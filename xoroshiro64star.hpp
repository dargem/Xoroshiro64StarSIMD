/* Written in 2016 by David Blackman and Sebastiano Vigna (vigna@acm.org).
 * SIMD implementation produced by Tristan Dyson.

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

// A note from the original authors

/* This is xoroshiro64* 1.0, our best and fastest 32-bit small-state
   generator for 32-bit floating-point numbers. We suggest to use its
   upper bits for floating-point generation, as it is slightly faster than
   xoroshiro64**. It passes all tests we are aware of except for linearity
   tests, as the lowest six bits have low linear complexity, so if low
   linear complexity is not considered an issue (as it is usually the
   case) it can be used to generate 32-bit outputs, too.

   We suggest to use a sign test to extract a random Boolean value, and
   right shifts to extract subsets of bits.

   The state must be seeded so that it is not everywhere zero. */

/* This has been modified heavily, using the same algorithm but implemented 
   in C++ using SIMD intrinsics to leverage CPU level parallelism. 
   This is suitable for generating large quantities of floats */

#include <cstdint>
#include <cstddef>
#include <array>
#include <cassert>
#include <immintrin.h>

enum class InstructionSet {
   AVX256,
   AVX512
};

// kinda hard coded can add more options later if more instructions are added
template <InstructionSet I>
static constexpr size_t RegisterByteSize = (I == InstructionSet::AVX256 ? 256 / 8 : 512 / 8);

// Change the defaulted instruction set to select it
template <InstructionSet I = InstructionSet::AVX512>
class alignas(RegisterByteSize<I>) XoroshiroRNG {
private:
   constexpr static size_t REGISTER_BYTE_SIZE = RegisterByteSize<I>;
public:
   constexpr static size_t ELEMENT_SIZE = sizeof(float);
   constexpr static size_t BATCH_SIZE = REGISTER_BYTE_SIZE / ELEMENT_SIZE;

   /**
    * @brief Get a batch of floats in an array container
    * The number of floats is the SIMD register size in bits / 32. 
    * AVX512 uses 512 bit registers, so batch size is 512/32 = 16.
    * @return std::array<double, BATCH_SIZE> 
    */
   std::array<float, BATCH_SIZE> getBatchFloats() {
      if constexpr (I == InstructionSet::AVX256) {
         __m256i avx_a = _mm256_load_epi32(a_states.data());
         __m256i avx_b = _mm256_load_epi32(b_states.data());
         __m256i mult = _mm256_set1_epi32(0x9E3779BB);
         const __m256i result = _mm256_mullo_epi32(avx_a, mult);

         avx_b = _mm256_xor_si256(avx_a, avx_b);
         avx_a = rotl(avx_a, 26);
         avx_a = _mm256_xor_si256(avx_a, avx_b);
         avx_a = _mm256_xor_si256(avx_a, _mm256_slli_epi32(avx_b, 9));

         avx_b = rotl(avx_b, 13);
         
         _mm256_store_epi32(a_states.data(), avx_a);
         _mm256_store_epi32(b_states, avx_b);

      }
      else if constexpr(I == InstructionSet::AVX512) {

      }
   };

   /**
    * @brief generate a batch of 8 floats
    * This is a wrapper around getFloatBatch() allowing a more explicit name at the callsite.
    * If you are not generating 8 floats in a batch this will compile error to warn you.
    * You are likely not using AVX256 which is the problem if you get a warning.
    * @return std::array<float, 8> container holding the random numbers
    */
   std::array<float, 8> get_eight_floats() requires (BATCH_SIZE == 8) {
      return getBatchFloats();
   }

   /**
    * @brief generate a batch of 16 floats
    * This is a wrapper around getFloatBatch() allowing a more explicit name at the callsite.
    * If you are not generating 16 floats in a batch this will compile error to warn you.
    * You are likely not using AVX512 which is the problem if you get a warning.
    * @return std::array<float, 16> container holding the random numbers
    */
   std::array<float, 16> get_sixteen_floats() requires (BATCH_SIZE == 16) {
      return getBatchFloats();
   }

private:
   // SOA style layout, BATCH_SIZE number of simultaneous RNGs, state split across 2 arrays
   alignas(REGISTER_BYTE_SIZE) std::array<uint32_t, BATCH_SIZE> a_states;
   alignas(REGISTER_BYTE_SIZE) std::array<uint32_t, BATCH_SIZE> b_states;

   __m256i rotl(__m256i x, const int k) {
      return _mm256_or_si256(
         _mm256_slli_epi32(x, k), 
         _mm256_srli_epi32(x, 32 - k)
      );
   }
};


static inline uint32_t rotl(const uint32_t x, int k) {
	return (x << k) | (x >> (32 - k));
}



static uint32_t s[2];

uint32_t next(void) {
	const uint32_t s0 = s[0];
	uint32_t s1 = s[1];
	const uint32_t result = s0 * 0x9E3779BB;


	s1 ^= s0;
	s[0] = rotl(s0, 26) ^ s1 ^ (s1 << 9); // a, b
	s[1] = rotl(s1, 13); // c

	return result;
}