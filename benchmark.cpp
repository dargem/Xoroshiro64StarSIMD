#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

#include "xoroshiro64star.hpp"

namespace {

using Clock = std::chrono::steady_clock;

static inline uint32_t rotl_scalar(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

struct ScalarXoroshiro64Star {
    uint32_t s[2] = {123456789u, 987654321u};

    inline uint32_t next_u32() {
        const uint32_t s0 = s[0];
        uint32_t s1 = s[1];

        const uint32_t result = s0 * 0x9E3779BBu;

        s1 ^= s0;
        s[0] = rotl_scalar(s0, 26) ^ s1 ^ (s1 << 9);
        s[1] = rotl_scalar(s1, 13);

        return result;
    }
};

static uint64_t parse_u64_or(const char* s, uint64_t fallback) {
    if (s == nullptr || *s == '\0') return fallback;
    char* end = nullptr;
    const auto v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') return fallback;
    return static_cast<uint64_t>(v);
}

struct BenchResult {
    uint64_t sum = 0;
    double seconds = 0.0;
    uint64_t count = 0;
};

template <class F>
BenchResult bench(std::string_view name, uint64_t count, F&& f) {
    volatile uint64_t sink = 0;

    const auto start = Clock::now();
    const uint64_t sum = f();
    const auto end = Clock::now();

    sink ^= sum;

    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count();

    std::cout << std::left << std::setw(18) << name << ": "
              << std::right << std::fixed << std::setprecision(6) << seconds << " s"
              << "  (" << std::setprecision(2) << (static_cast<double>(count) / seconds / 1e6) << " M u32/s)"
              << "  sum=" << sum
              << "\n";

    return BenchResult{.sum = static_cast<uint64_t>(sink), .seconds = seconds, .count = count};
}

} // namespace

int main(int argc, char** argv) {
    const uint64_t count = (argc >= 2) ? parse_u64_or(argv[1], 100000000ull) : 100000000ull;

    std::cout << "Benchmark generating " << count << " uint32 values\n";
    std::cout << "(Compile with -O3 and AVX2 enabled for SIMD path, e.g. -mavx2)\n\n";

    ScalarXoroshiro64Star scalar;

    XoroshiroRNG simd;
    constexpr size_t kBatch = decltype(simd)::BATCH_SIZE;

    const auto scalarResult = bench("scalar", count, [&] {
        uint64_t sum = 0;
        for (uint64_t i = 0; i < count; ++i) {
            sum += scalar.next_u32();
        }
        return sum;
    });

    const auto simdResult = bench("simd(batch)", count, [&] {
        uint64_t sum = 0;

        const uint64_t fullBatches = count / kBatch;
        const uint64_t remainder = count % kBatch;

        for (uint64_t i = 0; i < fullBatches; ++i) {
            const auto v = simd.get_batch_uint32();
            for (size_t j = 0; j < v.size(); ++j) {
                sum += v[j];
            }
        }

        if (remainder != 0) {
            const auto v = simd.get_batch_uint32();
            for (uint64_t j = 0; j < remainder; ++j) {
                sum += v[static_cast<size_t>(j)];
            }
        }

        return sum;
    });

    (void)scalarResult;
    (void)simdResult;

    return 0;
}
