#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>
#include <tuple>
#include <utility>
#include "xoroshiro64star.hpp"
#include <random>

#define NUM_XOR 20000000
#define NUM_ARRAY 2000000

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

struct BenchResult {
    std::string_view name;
    double seconds = 0.0;
    uint64_t count = 0;
};

template <size_t I, typename Tuple>
inline uint64_t invoke_at(Tuple& funcs) {
    return std::get<I>(funcs)();
}

template <typename Tuple, size_t... Is>
inline uint64_t invoke_by_index(size_t idx, Tuple& funcs, std::index_sequence<Is...>) {
    uint64_t result = 0;
    const bool matched = ((idx == Is ? (result = invoke_at<Is>(funcs), true) : false) || ...);
    (void)matched;
    return result;
}

template <size_t N, typename... Fs>
void bench(uint64_t count, const std::array<std::string_view, N>& names, size_t rounds, Fs&&... fs) {
    static_assert(N == sizeof...(Fs), "names must match number of benchmark functions");

    std::array<BenchResult, N> results{};
    for (size_t i = 0; i < N; ++i) {
        results[i].name = names[i];
    }

    auto funcs = std::tuple<Fs...>(std::forward<Fs>(fs)...);
    volatile uint64_t sink = 0;

    for (size_t round = 0; round < rounds; ++round) {
        for (size_t i = 0; i < N; ++i) {
            const auto start = Clock::now();
            asm volatile("" ::: "memory");
            const uint64_t result = invoke_by_index(i, funcs, std::make_index_sequence<N>{});
            asm volatile("" ::: "memory");
            const auto end = Clock::now();

            sink ^= result;

            const std::chrono::duration<double> elapsed = end - start;
            results[i].seconds += elapsed.count();
            results[i].count += count;
        }
    }

    for (const BenchResult& result : results) {
        std::cout << std::left << std::setw(18) << result.name << ": "
                << std::right << std::fixed << std::setprecision(6) << result.seconds << " s"
                << "  (" << std::setprecision(2) << (static_cast<double>(result.count) / result.seconds / 1e6) << " M u32/s)"
                << "\n";
    }
    (void)sink;
}

} // namespace

int main(int argc, char** argv) {

    // Warm up period
    volatile uint32_t sink{0};
    for (size_t i{}; i < 100000; ++i) {
        sink *= (sink >> 5);
    }

    ScalarXoroshiro64Star scalar;

    XoroshiroRNG simd;

    SequentialXoroshiroRNG<uint32_t> sequential;

    std::mt19937 mersenne(123u);
    std::uniform_int_distribution<uint32_t> uint_dist{};

    constexpr size_t kBatch = decltype(simd)::BATCH_SIZE;

    auto scalarXOR = [&] {
        uint64_t checksum = 0;
        for (uint64_t i = 0; i < NUM_XOR; ++i) {
            checksum ^= static_cast<uint64_t>(scalar.next_u32());
        }
        return checksum;
    };

    auto sequentialXOR = [&] {
        uint64_t checksum = 0;
        for (uint64_t i = 0; i < NUM_XOR; ++i) {
            checksum ^= static_cast<uint64_t>(sequential.get_uint32());
        }
        return checksum;
    };

    auto simdXOR = [&] {
        uint64_t checksum = 0;

        const uint64_t fullBatches = NUM_XOR / kBatch;
        const uint64_t remainder = NUM_XOR % kBatch;

#if defined(__AVX512F__)
            __m512i acc = _mm512_setzero_si512();
            for (uint64_t i = 0; i < fullBatches; ++i) {
                acc = _mm512_xor_si512(acc, simd.get_batch_uint32_simd());
            }
            alignas(64) std::array<uint32_t, 16> tmp{};
            _mm512_store_si512(reinterpret_cast<void*>(tmp.data()), acc);
            for (uint32_t x : tmp) checksum ^= static_cast<uint64_t>(x);
#elif defined(__AVX2__)
            __m256i acc = _mm256_setzero_si256();
            for (uint64_t i = 0; i < fullBatches; ++i) {
                acc = _mm256_xor_si256(acc, simd.get_batch_uint32_simd());
            }
            alignas(32) std::array<uint32_t, 8> tmp{};
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp.data()), acc);
            for (uint32_t x : tmp) checksum ^= static_cast<uint64_t>(x);
#elif defined(__AVX__)
            __m128i acc = _mm_setzero_si128();
            for (uint64_t i = 0; i < fullBatches; ++i) {
                acc = _mm_xor_si128(acc, simd.get_batch_uint32_simd());
            }
            alignas(16) std::array<uint32_t, 4> tmp{};
            _mm_store_si128(reinterpret_cast<__m128i*>(tmp.data()), acc);
            for (uint32_t x : tmp) checksum ^= static_cast<uint64_t>(x);
#else
            for (uint64_t i = 0; i < fullBatches; ++i) {
                const auto v = simd.get_batch_uint32();
                for (uint32_t x : v) checksum ^= static_cast<uint64_t>(x);
            }
#endif
            if (remainder != 0) {
                const auto v = simd.get_batch_uint32();
                for (uint64_t j = 0; j < remainder; ++j) {
                    checksum ^= static_cast<uint64_t>(v[static_cast<size_t>(j)]);
                }
            }

        return checksum;
    };

    auto mersenneXOR = [&] {
        uint64_t checksum = 0;
        for (uint64_t i = 0; i < NUM_XOR; ++i) {
            checksum ^= static_cast<uint64_t>(uint_dist(mersenne));
        }
        return checksum;
    };

    constexpr size_t xor_rounds = 30;
    bench(NUM_XOR, std::array<std::string_view, 4>{"mersenne(xor)", "scalar(xor)", "sequential(xor)", "simd(xor)"}, xor_rounds, mersenneXOR, scalarXOR, sequentialXOR, simdXOR);
    std::cout << '\n';

    alignas(XoroshiroRNG::REGISTER_BYTE_SIZE) std::array<uint32_t, NUM_ARRAY> arr{};
    arr.fill(0); // Make sure all the memory is mapped before filling
    asm volatile("" ::: "memory");

    auto scalarFill = [&] {
        for (size_t i{}; i < NUM_ARRAY; ++i) {
            arr[i] = scalar.next_u32();
        }
        return 0;
    };

    auto simdFill = [&]{
        simd.fill_aligned_uint32(arr.data(), NUM_ARRAY);
        return 0;
    };

    auto mersenneFill = [&] {
        for (size_t i{}; i < NUM_ARRAY; ++i) {
            arr[i] = uint_dist(mersenne);
        }
        return 0;
    };
    
    return 0;
}
