//! C ABI over rs-gigatoken for GraphTok's CPU route.
//!
//!   gtokcpu_create(tokenizer_json, cache_bytes) -> handle | null
//!   gtokcpu_encode(handle, text, len, &ids, &n)  -> 0 | -1
//!   gtokcpu_destroy(handle)
//!   gtokcpu_last_error()                         -> thread-local message
//!
//! One handle is one single-threaded encoder with its own pre-token cache,
//! forked from a process-wide prototype per tokenizer.json, so N handles share
//! the model tables but not the caches. A handle must not be used by two
//! threads at once (the C++ router holds a lock around it). The ids returned
//! by gtokcpu_encode stay valid until the next call on the same handle.
//! Every export catches panics; a panic reports -1 and the caller falls back
//! to the GPU route.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, OnceLock};

use rs_gigatoken::Tokenizer;

pub struct Handle {
    tok: Tokenizer,
    ids: Vec<u32>,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: impl Into<String>) {
    let msg = CString::new(msg.into().replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = msg);
}

fn protos() -> &'static Mutex<HashMap<String, Tokenizer>> {
    static P: OnceLock<Mutex<HashMap<String, Tokenizer>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(HashMap::new()))
}

fn create(path: &str, cache_bytes: u64) -> Result<Handle, String> {
    let mut map = protos().lock().map_err(|_| "prototype registry poisoned".to_string())?;
    if !map.contains_key(path) {
        let proto = rs_gigatoken::load_tokenizer::hf::load_hf_bpe(path)
            .map_err(|e| format!("gigatoken cannot load {path}: {e:#}"))?;
        map.insert(path.to_string(), proto);
    }
    let mut tok = map[path].fork();
    drop(map);
    if cache_bytes > 0 {
        tok.set_max_cache_bytes(Some(cache_bytes as usize));
    }
    Ok(Handle { tok, ids: Vec::new() })
}

/// Returns a new handle, or null with gtokcpu_last_error() set.
/// `cache_bytes` bounds the pre-token cache (0 = the crate default).
#[unsafe(no_mangle)]
pub extern "C" fn gtokcpu_create(tokenizer_json: *const c_char, cache_bytes: u64) -> *mut Handle {
    let r = catch_unwind(|| {
        if tokenizer_json.is_null() {
            return Err("null tokenizer path".to_string());
        }
        let path = unsafe { CStr::from_ptr(tokenizer_json) }
            .to_str()
            .map_err(|_| "tokenizer path is not UTF-8".to_string())?;
        create(path, cache_bytes)
    });
    match r {
        Ok(Ok(h)) => Box::into_raw(Box::new(h)),
        Ok(Err(e)) => {
            set_error(e);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_error("panic in gtokcpu_create");
            std::ptr::null_mut()
        }
    }
}

/// Encodes `len` bytes (raw text; added tokens and NFC are handled here) and
/// points `*ids` / `*n` at the handle-owned result. 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn gtokcpu_encode(
    h: *mut Handle,
    text: *const u8,
    len: usize,
    ids: *mut *const u32,
    n: *mut usize,
) -> i32 {
    if h.is_null() || ids.is_null() || n.is_null() || (text.is_null() && len > 0) {
        set_error("null argument");
        return -1;
    }
    let h = unsafe { &mut *h };
    let bytes: &[u8] = if len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(text, len) } };
    let r = catch_unwind(AssertUnwindSafe(|| {
        h.ids.clear();
        h.tok.encode_with_added_tokens_flat(bytes, &mut h.ids);
    }));
    match r {
        Ok(()) => {
            unsafe {
                *ids = h.ids.as_ptr();
                *n = h.ids.len();
            }
            0
        }
        Err(_) => {
            // The encoder may be mid-update; drop its caches by re-forking is
            // not possible without the prototype path, so poison the handle.
            h.ids.clear();
            set_error("panic in gtokcpu_encode");
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn gtokcpu_destroy(h: *mut Handle) {
    if !h.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(h) })));
    }
}

/// Message for the last failure on this thread ("" if none). Valid until the
/// next failing call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn gtokcpu_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}
