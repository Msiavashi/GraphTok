// The CPU route's content classifier (cpu_route.cc) must count exactly the
// same non-punctuation UTF-8 lead bytes on its AVX2 path as on its scalar
// path, on every length (vector body + scalar tail + the pair that spans
// them). Host-only; no GPU or data files.
#include "cpu_route.cc"

#include <cstdio>
#include <random>

int main() {
    std::mt19937 rng(1);
    for (int t = 0; t < 200000; ++t) {
        const size_t n = rng() % 300;
        std::vector<unsigned char> b(n);
        for (auto& x : b) {
            const unsigned r = rng() % 8;
            x = r == 0 ? 0xE2 : r == 1 ? 0x80 + rng() % 2 : r == 2 ? 0xC0 + rng() % 64 : rng() % 256;
        }
        const size_t s = gbpe::nonpunct_leads_scalar(b.data(), n);
        const size_t v = gbpe::nonpunct_leads_avx2(b.data(), n);
        if (s != v) {
            std::printf("FAIL n=%zu scalar=%zu avx2=%zu\n", n, s, v);
            return 1;
        }
    }
    std::puts("PASS classifier: scalar == avx2 on 200000 random buffers");
    return 0;
}
