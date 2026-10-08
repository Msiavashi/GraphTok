# Third-party code in the gigatoken CPU engine

Builds that link the CPU engine (`-DGBPE_GIGATOKEN=AUTO|ON` with a nightly
cargo) statically include rs-gigatoken 0.10.0 (vendored in
`third_party/rs-gigatoken`) and the Rust crates below. Regenerate this list
with `cargo metadata --locked` in `rust/gtok_cpu` after any dependency change.
Where a crate offers a choice of licenses, GraphTok uses it under the first
permissive option listed (MIT, Apache-2.0, BSD, ISC, Zlib, Unicode-3.0, MPL-2.0).

## rs-gigatoken

```
MIT License

Copyright (c) 2026 Marcel Rød

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Linked crates (name, version, license)

| crate | version | license |
|---|---|---|
| adler2 | 2.0.1 | 0BSD OR MIT OR Apache-2.0 |
| ahash | 0.8.12 | MIT OR Apache-2.0 |
| aho-corasick | 1.1.5 | Unlicense OR MIT |
| alloc-no-stdlib | 2.0.4 | BSD-3-Clause |
| alloc-stdlib | 0.2.4 | BSD-3-Clause |
| android_system_properties | 0.1.5 | MIT/Apache-2.0 |
| arrow-array | 59.3.0 | Apache-2.0 AND MIT |
| arrow-buffer | 59.3.0 | Apache-2.0 |
| arrow-data | 59.3.0 | Apache-2.0 |
| arrow-ipc | 59.3.0 | Apache-2.0 |
| arrow-schema | 59.3.0 | Apache-2.0 |
| arrow-select | 59.3.0 | Apache-2.0 |
| autocfg | 1.5.1 | Apache-2.0 OR MIT |
| base64 | 0.13.1 | MIT/Apache-2.0 |
| base64 | 0.22.1 | MIT OR Apache-2.0 |
| base64 | 0.23.1 | MIT OR Apache-2.0 |
| bitflags | 2.13.2 | MIT OR Apache-2.0 |
| brotli | 8.0.4 | BSD-3-Clause AND MIT |
| brotli-decompressor | 5.0.3 | BSD-3-Clause/MIT |
| bumpalo | 3.20.3 | MIT OR Apache-2.0 |
| bytes | 1.12.1 | MIT |
| calendrical_calculations | 0.2.4 | Apache-2.0 |
| cc | 1.4.7 | MIT OR Apache-2.0 |
| cfg-if | 1.0.5 | MIT OR Apache-2.0 |
| chrono | 0.4.45 | MIT OR Apache-2.0 |
| cobs | 0.3.0 | MIT OR Apache-2.0 |
| console | 0.16.6 | MIT |
| const-random | 0.1.18 | MIT OR Apache-2.0 |
| const-random-macro | 0.1.16 | MIT OR Apache-2.0 |
| core-foundation-sys | 0.8.7 | MIT OR Apache-2.0 |
| core_maths | 0.1.1 | MIT |
| crc32fast | 1.5.2 | MIT OR Apache-2.0 |
| crossbeam-deque | 0.8.8 | MIT OR Apache-2.0 |
| crossbeam-epoch | 0.9.21 | MIT OR Apache-2.0 |
| crossbeam-utils | 0.8.23 | MIT OR Apache-2.0 |
| crunchy | 0.2.4 | MIT |
| dashmap | 6.2.1 | MIT |
| databake | 0.2.1 | Unicode-3.0 |
| databake-derive | 0.2.2 | Unicode-3.0 |
| displaydoc | 0.2.7 | MIT OR Apache-2.0 |
| either | 1.18.0 | MIT OR Apache-2.0 |
| encode_unicode | 1.0.0 | Apache-2.0 OR MIT |
| equivalent | 1.0.2 | Apache-2.0 OR MIT |
| erased-serde | 0.4.10 | MIT OR Apache-2.0 |
| eyre | 0.6.14 | MIT OR Apache-2.0 |
| faststr | 0.2.34 | MIT OR Apache-2.0 |
| find-msvc-tools | 0.1.13 | MIT OR Apache-2.0 |
| fixed_decimal | 0.7.2 | Unicode-3.0 |
| flatbuffers | 25.12.19 | Apache-2.0 |
| flate2 | 1.1.10 | MIT OR Apache-2.0 |
| futures-core | 0.3.34 | MIT OR Apache-2.0 |
| futures-task | 0.3.34 | MIT OR Apache-2.0 |
| futures-util | 0.3.34 | MIT OR Apache-2.0 |
| getrandom | 0.2.17 | MIT OR Apache-2.0 |
| getrandom | 0.3.4 | MIT OR Apache-2.0 |
| getrandom | 0.4.3 | MIT OR Apache-2.0 |
| half | 2.7.1 | MIT OR Apache-2.0 |
| hashbrown | 0.14.5 | MIT OR Apache-2.0 |
| hashbrown | 0.17.1 | MIT OR Apache-2.0 |
| http | 1.5.0 | MIT OR Apache-2.0 |
| httparse | 1.10.1 | MIT OR Apache-2.0 |
| iana-time-zone | 0.1.65 | MIT OR Apache-2.0 |
| iana-time-zone-haiku | 0.1.2 | MIT OR Apache-2.0 |
| icu | 2.3.1 | Unicode-3.0 |
| icu_calendar | 2.3.0 | Unicode-3.0 |
| icu_calendar_data | 2.3.0 | Unicode-3.0 |
| icu_casemap | 2.3.0 | Unicode-3.0 |
| icu_casemap_data | 2.3.0 | Unicode-3.0 |
| icu_collator | 2.3.1 | Unicode-3.0 |
| icu_collator_data | 2.3.0 | Unicode-3.0 |
| icu_collections | 2.3.0 | Unicode-3.0 |
| icu_datetime | 2.3.0 | Unicode-3.0 |
| icu_datetime_data | 2.3.0 | Unicode-3.0 |
| icu_decimal | 2.3.0 | Unicode-3.0 |
| icu_decimal_data | 2.3.0 | Unicode-3.0 |
| icu_experimental | 0.6.0 | Unicode-3.0 |
| icu_experimental_data | 0.6.0 | Unicode-3.0 |
| icu_list | 2.3.0 | Unicode-3.0 |
| icu_list_data | 2.3.0 | Unicode-3.0 |
| icu_locale | 2.3.1 | Unicode-3.0 |
| icu_locale_core | 2.3.0 | Unicode-3.0 |
| icu_locale_data | 2.3.0 | Unicode-3.0 |
| icu_locale_fallback | 2.3.0 | Unicode-3.0 |
| icu_locale_fallback_data | 2.3.0 | Unicode-3.0 |
| icu_normalizer | 2.3.0 | Unicode-3.0 |
| icu_normalizer_data | 2.3.0 | Unicode-3.0 |
| icu_pattern | 0.5.0 | Unicode-3.0 |
| icu_plurals | 2.3.0 | Unicode-3.0 |
| icu_plurals_data | 2.3.0 | Unicode-3.0 |
| icu_properties | 2.3.0 | Unicode-3.0 |
| icu_properties_data | 2.3.0 | Unicode-3.0 |
| icu_provider | 2.3.1 | Unicode-3.0 |
| icu_provider_registry | 2.3.0 | Unicode-3.0 |
| icu_segmenter | 2.3.0 | Unicode-3.0 |
| icu_segmenter_data | 2.3.0 | Unicode-3.0 |
| icu_time | 2.3.0 | Unicode-3.0 |
| icu_time_data | 2.3.1 | Unicode-3.0 |
| indenter | 0.3.4 | MIT OR Apache-2.0 |
| indexmap | 2.14.2 | Apache-2.0 OR MIT |
| indicatif | 0.18.6 | MIT |
| itertools | 0.15.0 | MIT OR Apache-2.0 |
| itoa | 1.0.18 | MIT OR Apache-2.0 |
| ixdtf | 0.6.6 | Unicode-3.0 |
| jobserver | 0.1.35 | MIT OR Apache-2.0 |
| js-sys | 0.3.104 | MIT OR Apache-2.0 |
| libc | 0.2.189 | MIT OR Apache-2.0 |
| libm | 0.2.16 | MIT |
| litemap | 0.8.3 | Unicode-3.0 |
| lock_api | 0.4.14 | MIT OR Apache-2.0 |
| log | 0.4.34 | MIT OR Apache-2.0 |
| lz4_flex | 0.14.0 | MIT |
| memchr | 2.8.3 | Unlicense OR MIT |
| memmap2 | 0.9.11 | MIT OR Apache-2.0 |
| minimal-lexical | 0.2.1 | MIT/Apache-2.0 |
| miniz_oxide | 0.9.1 | MIT OR Zlib OR Apache-2.0 |
| munge | 0.4.7 | MIT |
| munge_macro | 0.4.7 | MIT |
| nom | 7.1.3 | MIT |
| num-bigint | 0.4.8 | MIT OR Apache-2.0 |
| num-bigint | 0.5.1 | MIT OR Apache-2.0 |
| num-complex | 0.4.6 | MIT OR Apache-2.0 |
| num-integer | 0.1.47 | MIT OR Apache-2.0 |
| num-rational | 0.4.2 | MIT OR Apache-2.0 |
| num-traits | 0.2.19 | MIT OR Apache-2.0 |
| once_cell | 1.21.4 | MIT OR Apache-2.0 |
| parking_lot_core | 0.9.12 | MIT OR Apache-2.0 |
| parquet | 59.3.0 | Apache-2.0 |
| percent-encoding | 2.3.2 | MIT OR Apache-2.0 |
| pin-project-lite | 0.2.17 | Apache-2.0 OR MIT |
| pkg-config | 0.3.34 | MIT OR Apache-2.0 |
| portable-atomic | 1.15.0 | Apache-2.0 OR MIT |
| postcard | 1.1.3 | MIT OR Apache-2.0 |
| potential_utf | 0.1.6 | Unicode-3.0 |
| priority-queue | 2.7.0 | LGPL-3.0-or-later OR MPL-2.0 |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 |
| ptr_meta | 0.3.2 | MIT |
| ptr_meta_derive | 0.3.2 | MIT |
| quote | 1.0.47 | MIT OR Apache-2.0 |
| r-efi | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later |
| r-efi | 6.0.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later |
| rancor | 0.1.3 | MIT |
| rayon | 1.12.0 | MIT OR Apache-2.0 |
| rayon-core | 1.13.0 | MIT OR Apache-2.0 |
| redox_syscall | 0.5.18 | MIT |
| ref-cast | 1.0.27 | MIT OR Apache-2.0 |
| ref-cast-impl | 1.0.27 | MIT OR Apache-2.0 |
| regex-automata | 0.4.18 | MIT OR Apache-2.0 |
| regex-syntax | 0.8.11 | MIT OR Apache-2.0 |
| rend | 0.5.4 | MIT |
| ring | 0.17.14 | Apache-2.0 AND ISC |
| rkyv | 0.8.18 | MIT |
| rkyv_derive | 0.8.18 | MIT |
| rs-gigatoken | 0.10.0 | MIT |
| rustc-hash | 2.1.3 | Apache-2.0 OR MIT |
| rustc_version | 0.4.1 | MIT OR Apache-2.0 |
| rustls | 0.23.45 | Apache-2.0 OR ISC OR MIT |
| rustls-pki-types | 1.15.1 | MIT OR Apache-2.0 |
| rustls-webpki | 0.103.15 | ISC |
| rustversion | 1.0.23 | MIT OR Apache-2.0 |
| scopeguard | 1.2.0 | MIT OR Apache-2.0 |
| semver | 1.0.28 | MIT OR Apache-2.0 |
| seq-macro | 0.3.6 | MIT OR Apache-2.0 |
| serde | 1.0.229 | MIT OR Apache-2.0 |
| serde_core | 1.0.229 | MIT OR Apache-2.0 |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 |
| shlex | 2.0.1 | MIT OR Apache-2.0 |
| simd-adler32 | 0.3.10 | MIT |
| simdutf | 0.7.0 | MIT |
| simdutf8 | 0.1.5 | MIT OR Apache-2.0 |
| slab | 0.4.12 | MIT |
| smallvec | 1.16.1 | MIT OR Apache-2.0 |
| snap | 1.1.2 | BSD-3-Clause |
| sonic-number | 0.1.3 | Apache-2.0 |
| sonic-rs | 0.5.10 | Apache-2.0 |
| sonic-simd | 0.1.4 | Apache-2.0 |
| spm_precompiled | 0.1.4 | Apache-2.0 |
| stable_deref_trait | 1.2.1 | MIT OR Apache-2.0 |
| subtle | 2.6.1 | BSD-3-Clause |
| syn | 2.0.119 | MIT OR Apache-2.0 |
| syn | 3.0.6 | MIT OR Apache-2.0 |
| synstructure | 0.14.0 | MIT |
| thiserror | 2.0.20 | MIT OR Apache-2.0 |
| thiserror-impl | 2.0.20 | MIT OR Apache-2.0 |
| tiny-keccak | 2.0.2 | CC0-1.0 |
| tinystr | 0.8.4 | Unicode-3.0 |
| tinyvec | 1.13.3 | Zlib OR Apache-2.0 OR MIT |
| twox-hash | 2.1.4 | MIT |
| typeid | 1.0.3 | MIT OR Apache-2.0 |
| unicode-ident | 1.0.26 | (MIT OR Apache-2.0) AND Unicode-3.0 |
| unicode-segmentation | 1.13.3 | MIT OR Apache-2.0 |
| unicode-width | 0.2.2 | MIT OR Apache-2.0 |
| unit-prefix | 0.5.2 | MIT |
| untrusted | 0.9.0 | ISC |
| ureq | 3.4.2 | MIT OR Apache-2.0 |
| ureq-proto | 0.6.4 | MIT OR Apache-2.0 |
| utf16_iter | 1.0.5 | Apache-2.0 OR MIT |
| utf8-zero | 0.8.1 | MIT OR Apache-2.0 |
| utf8_iter | 1.0.4 | Apache-2.0 OR MIT |
| uuid | 1.26.1 | Apache-2.0 OR MIT |
| version_check | 0.9.5 | MIT/Apache-2.0 |
| wasi | 0.11.1+wasi-snapshot-preview1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wasip2 | 1.0.4+wasi-0.2.12 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wasm-bindgen | 0.2.127 | MIT OR Apache-2.0 |
| wasm-bindgen-macro | 0.2.127 | MIT OR Apache-2.0 |
| wasm-bindgen-macro-support | 0.2.127 | MIT OR Apache-2.0 |
| wasm-bindgen-shared | 0.2.127 | MIT OR Apache-2.0 |
| web-time | 1.1.0 | MIT OR Apache-2.0 |
| webpki-roots | 1.0.9 | CDLA-Permissive-2.0 |
| windows-core | 0.62.2 | MIT OR Apache-2.0 |
| windows-implement | 0.60.2 | MIT OR Apache-2.0 |
| windows-interface | 0.59.3 | MIT OR Apache-2.0 |
| windows-link | 0.2.1 | MIT OR Apache-2.0 |
| windows-result | 0.4.1 | MIT OR Apache-2.0 |
| windows-strings | 0.5.1 | MIT OR Apache-2.0 |
| windows-sys | 0.52.0 | MIT OR Apache-2.0 |
| windows-sys | 0.61.2 | MIT OR Apache-2.0 |
| windows-targets | 0.52.6 | MIT OR Apache-2.0 |
| windows_aarch64_gnullvm | 0.52.6 | MIT OR Apache-2.0 |
| windows_aarch64_msvc | 0.52.6 | MIT OR Apache-2.0 |
| windows_i686_gnu | 0.52.6 | MIT OR Apache-2.0 |
| windows_i686_gnullvm | 0.52.6 | MIT OR Apache-2.0 |
| windows_i686_msvc | 0.52.6 | MIT OR Apache-2.0 |
| windows_x86_64_gnu | 0.52.6 | MIT OR Apache-2.0 |
| windows_x86_64_gnullvm | 0.52.6 | MIT OR Apache-2.0 |
| windows_x86_64_msvc | 0.52.6 | MIT OR Apache-2.0 |
| winnow | 1.0.4 | MIT |
| wit-bindgen | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| write16 | 1.0.0 | Apache-2.0 OR MIT |
| writeable | 0.6.4 | Unicode-3.0 |
| yoke | 0.8.3 | Unicode-3.0 |
| yoke-derive | 0.8.3 | Unicode-3.0 |
| zerocopy | 0.8.57 | BSD-2-Clause OR Apache-2.0 OR MIT |
| zerocopy-derive | 0.8.57 | BSD-2-Clause OR Apache-2.0 OR MIT |
| zerofrom | 0.1.8 | Unicode-3.0 |
| zerofrom-derive | 0.1.8 | Unicode-3.0 |
| zeroize | 1.9.0 | Apache-2.0 OR MIT |
| zerotrie | 0.2.5 | Unicode-3.0 |
| zerovec | 0.11.8 | Unicode-3.0 |
| zerovec-derive | 0.11.6 | Unicode-3.0 |
| zlib-rs | 0.6.8 | Zlib |
| zmij | 1.0.23 | MIT |
| zstd | 0.13.3 | MIT |
| zstd-safe | 7.3.0 | BSD-3-Clause |
| zstd-sys | 2.1.0+zstd.1.5.7 | BSD-3-Clause |
