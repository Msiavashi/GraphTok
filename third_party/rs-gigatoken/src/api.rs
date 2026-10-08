//! High-level, Python-free convenience APIs built on the zero-copy core.
//!
//! These helpers intentionally stay outside the hot tokenization loops. They
//! provide the compatibility-oriented assembly operations that applications
//! commonly need after encoding: wrapping/truncation, padding, and compiled
//! special-token matching.

use aho_corasick::{AhoCorasick, MatchKind};
use eyre::{Result, eyre};
use std::borrow::Borrow;

/// Options for assembling a ragged batch into fixed-width rows.
#[derive(Debug, Clone)]
pub struct PadTruncate {
    /// Token ID used for padding.
    pub pad_id: u32,
    /// Maximum output width. Required when `truncate` is true.
    pub max_length: Option<usize>,
    /// Force every row to `max_length`; otherwise rows use the longest row.
    pub pad_to_max_length: bool,
    /// Truncate rows longer than `max_length`.
    pub truncate: bool,
    /// Put padding on the left instead of the right.
    pub pad_left: bool,
    /// Keep the rightmost tokens when truncating instead of the leftmost.
    pub truncate_left: bool,
    /// Token IDs written before every row. They count toward `max_length`.
    pub prefix: Vec<u32>,
    /// Token IDs written after every row. They count toward `max_length`.
    pub suffix: Vec<u32>,
}

/// Native Rust name for the encoding assembly options.
///
/// `PadTruncate` remains available for source compatibility with the earlier
/// Rust-only migration; new code can use `EncodeOptions`.
pub type EncodeOptions = PadTruncate;

impl PadTruncate {
    pub const fn new(pad_id: u32) -> Self {
        Self {
            pad_id,
            max_length: None,
            pad_to_max_length: false,
            truncate: false,
            pad_left: false,
            truncate_left: false,
            prefix: Vec::new(),
            suffix: Vec::new(),
        }
    }
}

/// Apply padding/truncation to a flat ragged representation.
///
/// `tokens` contains all rows concatenated and `lengths` contains one length
/// per row. The result is `(flat_matrix, width, output_lengths)`, where the
/// matrix is row-major and every row has exactly `width` entries.
pub fn pad_truncate_ragged<O>(
    tokens: &[u32],
    lengths: &[usize],
    options: O,
) -> Result<(Vec<u32>, usize, Vec<usize>)>
where
    O: Borrow<PadTruncate>,
{
    let options = options.borrow();
    let expected: usize = lengths.iter().sum();
    if expected != tokens.len() {
        return Err(eyre!(
            "ragged lengths sum to {expected}, but token buffer contains {} ids",
            tokens.len()
        ));
    }

    let extra = options.prefix.len() + options.suffix.len();
    let keep_limit = if options.truncate {
        let max = options
            .max_length
            .ok_or_else(|| eyre!("truncate requires max_length"))?;
        max.checked_sub(extra).ok_or_else(|| {
            eyre!(
                "max_length={max} leaves no room for {} special tokens added per sequence",
                extra
            )
        })?
    } else {
        usize::MAX
    };

    let longest_tokens = lengths.iter().copied().max().unwrap_or(0);
    let longest = longest_tokens.min(keep_limit) + extra;

    let width = if options.pad_to_max_length {
        let max = options
            .max_length
            .ok_or_else(|| eyre!("pad_to_max_length requires max_length"))?;
        if longest > max {
            return Err(eyre!(
                "a sequence is {longest} ids long but max_length={max} was requested without truncation"
            ));
        }
        max
    } else {
        longest
    };

    let mut out = vec![options.pad_id; lengths.len() * width];
    let mut out_lengths = Vec::with_capacity(lengths.len());
    let mut offset = 0usize;

    for (row, &len) in lengths.iter().enumerate() {
        let keep = len.min(keep_limit);
        let source_start = if options.truncate_left {
            offset + len - keep
        } else {
            offset
        };
        let row_len = extra + keep;
        let dst_start = if options.pad_left { width - row_len } else { 0 };
        let dst = &mut out[row * width + dst_start..row * width + dst_start + row_len];
        dst[..options.prefix.len()].copy_from_slice(&options.prefix);
        let token_start = options.prefix.len();
        dst[token_start..token_start + keep]
            .copy_from_slice(&tokens[source_start..source_start + keep]);
        let suffix_start = token_start + keep;
        dst[suffix_start..suffix_start + options.suffix.len()].copy_from_slice(&options.suffix);
        out_lengths.push(row_len);
        offset += len;
    }

    Ok((out, width, out_lengths))
}

/// A compiled multi-pattern matcher for special-token or sentinel scanning.
///
/// Patterns are compiled once and can then be reused across many documents.
/// Matching uses Aho-Corasick's leftmost-longest semantics, matching the
/// tokenizer engine's added-token scanner.
pub struct SubstringMatcher {
    automaton: AhoCorasick,
}

impl SubstringMatcher {
    /// Build a matcher. Empty patterns are rejected because they do not form a
    /// useful special-token boundary and would match at every byte position.
    pub fn new<'a, I>(patterns: I) -> Result<Self>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let patterns: Vec<&[u8]> = patterns.into_iter().collect();
        if patterns.iter().any(|p| p.is_empty()) {
            return Err(eyre!("empty matcher patterns are not supported"));
        }
        let automaton = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .map_err(|e| eyre!("failed to build matcher: {e}"))?;
        Ok(Self { automaton })
    }

    /// Return `(pattern_index, start, end)` for each non-overlapping match.
    pub fn find_iter(&self, haystack: &[u8]) -> impl Iterator<Item = (usize, usize, usize)> {
        self.automaton
            .find_iter(haystack)
            .map(|m| (m.pattern().as_usize(), m.start(), m.end()))
    }

    /// Return the first match, if any.
    pub fn find(&self, haystack: &[u8]) -> Option<(usize, usize, usize)> {
        self.automaton
            .find(haystack)
            .map(|m| (m.pattern().as_usize(), m.start(), m.end()))
    }

    /// Return whether any pattern occurs in `haystack`.
    #[inline]
    pub fn contains(&self, haystack: &[u8]) -> bool {
        self.automaton.is_match(haystack)
    }

    /// Collect all non-overlapping matches.
    pub fn find_all(&self, haystack: &[u8]) -> Vec<(usize, usize, usize)> {
        self.find_iter(haystack).collect()
    }
}

/// Add a prefix/suffix and optionally truncate a token row.
///
/// When `max_length` is set, truncation is applied to the original row before
/// adding the suffix; this keeps the wrapper deterministic and avoids an
/// accidental output larger than the requested budget.
pub fn wrap_truncate(
    row: &[u32],
    prefix: &[u32],
    suffix: &[u32],
    max_length: Option<usize>,
    truncate_left: bool,
) -> Vec<u32> {
    let available = max_length
        .map(|max| max.saturating_sub(prefix.len() + suffix.len()))
        .unwrap_or(row.len());
    let keep = row.len().min(available);
    let start = if truncate_left { row.len() - keep } else { 0 };

    let mut out = Vec::with_capacity(prefix.len() + keep + suffix.len());
    out.extend_from_slice(prefix);
    out.extend_from_slice(&row[start..start + keep]);
    out.extend_from_slice(suffix);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_right_and_left() {
        let tokens = [1, 2, 3, 4, 5];
        let lengths = [2, 3];
        let mut opts = PadTruncate::new(0);
        opts.max_length = Some(4);
        opts.pad_to_max_length = true;
        let (right, width, lens) = pad_truncate_ragged(&tokens, &lengths, &opts).unwrap();
        assert_eq!(width, 4);
        assert_eq!(lens, vec![2, 3]);
        assert_eq!(right, vec![1, 2, 0, 0, 3, 4, 5, 0]);

        opts.pad_left = true;
        let (left, _, _) = pad_truncate_ragged(&tokens, &lengths, &opts).unwrap();
        assert_eq!(left, vec![0, 0, 1, 2, 0, 3, 4, 5]);
    }

    #[test]
    fn prefix_suffix_count_toward_truncation_budget() {
        let mut opts = PadTruncate::new(0);
        opts.max_length = Some(5);
        opts.pad_to_max_length = true;
        opts.truncate = true;
        opts.prefix = vec![9];
        opts.suffix = vec![8];
        let (out, width, lens) = pad_truncate_ragged(&[1, 2, 3, 4], &[4], &opts).unwrap();
        assert_eq!(width, 5);
        assert_eq!(lens, vec![5]);
        assert_eq!(out, vec![9, 1, 2, 3, 8]);
    }

    #[test]
    fn truncate_keeps_requested_side() {
        let mut opts = PadTruncate::new(0);
        opts.max_length = Some(2);
        opts.truncate = true;
        let (left, _, _) = pad_truncate_ragged(&[1, 2, 3], &[3], &opts).unwrap();
        assert_eq!(left, vec![1, 2]);
        opts.truncate_left = true;
        let (right, _, _) = pad_truncate_ragged(&[1, 2, 3], &[3], &opts).unwrap();
        assert_eq!(right, vec![2, 3]);
    }

    #[test]
    fn matcher_is_leftmost_longest() {
        let matcher = SubstringMatcher::new([b"<a>".as_slice(), b"<ab>".as_slice()]).unwrap();
        assert_eq!(matcher.find(b"x<ab>y"), Some((1, 1, 5)));
    }

    #[test]
    fn wrapping_respects_budget() {
        assert_eq!(
            wrap_truncate(&[1, 2, 3], &[9], &[8], Some(3), false),
            vec![9, 1, 8]
        );
        assert_eq!(
            wrap_truncate(&[1, 2, 3], &[9], &[8], Some(3), true),
            vec![9, 3, 8]
        );
    }
}
