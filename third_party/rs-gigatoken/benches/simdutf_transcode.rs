use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rayon::prelude::*;
use std::hint::black_box;

const TARGET_BENCH_SIZE: usize = 100_000_000; // ~100 MB

/// Load OWT data, truncated to a UTF-8-safe boundary near `max_bytes`.
fn load_owt(max_bytes: usize) -> Option<Vec<u8>> {
    let data_dir = std::env::home_dir().unwrap().join("data");
    let owt_path = std::env::home_dir().unwrap().join("data/owt_train.txt");

    if !owt_path.exists() {
        eprintln!("Skipping benchmark: OpenWebText dataset not found at {owt_path:?}.");

        eprintln!("Install ~/data/owt_train.txt to run the benchmark.");

        return None;
    }

    let all_bytes = match std::fs::read(data_dir.join("owt_train.txt")) {
        Ok(data) => data,

        Err(err) => {
            eprintln!("Skipping benchmark: could not read {owt_path:?}: {err}");

            return None;
        }
    };
    let mut end = max_bytes.min(all_bytes.len());
    while end > 0 && std::str::from_utf8(&all_bytes[..end]).is_err() {
        end -= 1;
    }
    Some(all_bytes[..end].to_vec())
}

fn simdutf_transcode_benches(c: &mut Criterion) {
    let Some(input) = load_owt(TARGET_BENCH_SIZE) else {
        return;
    };
    let input_len = input.len() as u64;
    eprintln!("Benchmark input size: {:.1} MB", input_len as f64 / 1e6);

    let mut group = c.benchmark_group("simdutf_transcode");
    group.throughput(Throughput::Bytes(input_len));
    group.sample_size(10);

    for num_threads in [1, 2, 4, 8, 12, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let chunk_size = input.len() / num_threads;

        // Pre-compute chunk boundaries aligned to UTF-8 char boundaries.
        let mut boundaries = Vec::with_capacity(num_threads + 1);
        boundaries.push(0);
        for i in 1..num_threads {
            let mut pos = chunk_size * i;
            while pos < input.len() && input[pos] & 0b1100_0000 == 0b1000_0000 {
                pos += 1;
            }
            boundaries.push(pos);
        }
        boundaries.push(input.len());

        // Pre-allocate one destination buffer per thread.
        let mut dst_bufs: Vec<Vec<u32>> = boundaries
            .windows(2)
            .map(|w| vec![0u32; simdutf::utf32_length_from_utf8(&input[w[0]..w[1]])])
            .collect();

        group.bench_function(format!("utf8_to_utf32_{num_threads}t"), |b| {
            b.iter(|| {
                pool.install(|| {
                    let total: usize = boundaries
                        .par_windows(2)
                        .zip(&mut dst_bufs)
                        .map(|(w, dst)| {
                            let chunk = &input[w[0]..w[1]];
                            let written = unsafe {
                                simdutf::convert_valid_utf8_to_utf32(
                                    chunk.as_ptr(),
                                    chunk.len(),
                                    dst.as_mut_ptr(),
                                )
                            };
                            black_box(written)
                        })
                        .sum();
                    black_box(total);
                })
            });
        });
    }

    group.finish();
}

criterion_group!(benches, simdutf_transcode_benches);
criterion_main!(benches);
