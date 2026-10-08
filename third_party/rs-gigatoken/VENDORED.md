# rs-gigatoken 0.10.0 (vendored)

Source: crates.io `rs-gigatoken` 0.10.0 (MIT, Copyright (c) 2026 Marcel Rød),
repository https://github.com/InfidelRahul/rs-gigatoken. `src/` is
byte-identical to the published crate. The only change is `Cargo.toml`: the
`[profile.*]` tables were removed, because the published manifest uses the
nightly-only `profile-rustflags` Cargo feature, which current nightly Cargo
rejects; dependency profiles are ignored by dependents anyway.

GraphTok links it through `rust/gtok_cpu` as its CPU route for small inputs.
