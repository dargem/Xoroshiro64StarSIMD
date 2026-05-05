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
#include <random>
#include <limits>

enum class InstructionSet {
   AVX256,
   AVX512
};

template <InstructionSet I>
struct InstructionSetTraits;

template<>
struct InstructionSetTraits<InstructionSet::AVX256> {
   static constexpr size_t bits = 256;

   using __mi = __m256i;
   using __m = __m256;

   // Integer ops
   static __mi _mm_load_si(__mi const* mem_addr) {
      return _mm256_load_si256(mem_addr);
   }

   static __mi _mm_srli_epi32(__mi a, int bits) {
      return _mm256_srli_epi32(a, bits);
   }

   static __mi _mm_slli_epi32(__mi a, int bits) {
      return _mm256_slli_epi32(a, bits);
   }

   static __mi _mm_set1_epi32(int val) {
      return _mm256_set1_epi32(val);
   }

   static __mi _mm_mullo_epi32(__mi a, __mi b) {
      return _mm256_mullo_epi32(a, b);
   }

   static __mi _mm_xor_si(__mi a, __mi b) {
      return _mm256_xor_si256(a, b);
   }

   static __mi _mm_or_si(__mi a, __mi b) {
      return _mm256_or_si256(a, b);
   }

   static void _mm_store_si(__mi* mem_addr, __mi source) {
      _mm256_store_si256(mem_addr, source);
   }


   // Float ops
   static __m _mm_sub_ps(__m a, __m b) {
      return _mm256_sub_ps(a, b);
   }

   static __m _mm_set1_ps(float val) {
      return _mm256_set1_ps(val);
   }

   static __m _mm_castsi_ps(__mi a) {
      return _mm256_castsi256_ps(a);
   }


};

template <>
struct InstructionSetTraits<InstructionSet::AVX512> {
    static constexpr size_t bits = 512;
};

template <InstructionSet I>
inline constexpr size_t RegisterBitSize = InstructionSetTraits<I>::bits;

template <InstructionSet I>
inline constexpr size_t RegisterByteSize = RegisterBitSize<I> / 8;

// Change the defaulted instruction set to select it
template <InstructionSet I = InstructionSet::AVX256>
class alignas(RegisterByteSize<I>) XoroshiroRNG {
private:
   constexpr static size_t REGISTER_BYTE_SIZE = RegisterByteSize<I>;
   using S = InstructionSetTraits<I>;
   using __m = S::__m;
   using __mi = S::__mi;
public:
   constexpr static size_t ELEMENT_SIZE = sizeof(float);
   constexpr static size_t BATCH_SIZE = REGISTER_BYTE_SIZE / ELEMENT_SIZE;

   XoroshiroRNG() {
      // do something actually correct later just slop this for now
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<uint32_t> distr(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());

      for (uint32_t& a : a_states) { a = distr(gen); }
      for (uint32_t& b : b_states) { b = distr(gen); }
   }

   /**
    * @brief Get a batch of floats in an array container
    * The number of floats is the SIMD register size in bits / 32. 
    * AVX512 uses 512 bit registers, so batch size is 512/32 = 16.
    * @return std::array<float, BATCH_SIZE> 
    */
   [[nodiscard]]
   std::array<float, BATCH_SIZE> getBatchFloats() {
      if constexpr (I == InstructionSet::AVX256) {

         __mi result = advance();

         // Need to convert result into a [0, 1) float
         // bit shift 9 to the right to get rid of sign + 8 bit exponent
         result = S::_mm_srli_epi32(result, 9);

         // want this to be converted to [0, 1], want a leading 0 so its signed positive, 
         // for a floating point we know number = 2^n * (1 + mantissa), where mantissa = [0, 1)
         // if we have 2^n = 1, then number = 1 + mantissa = [1, 2), 
         // therefore [0, 1) = [1, 2) - 1 = 2^0 * (1 + mantissa)
         // want n to equal 0, but n is found through subtracting exponent field by 127 in a float
         // so we want exponent field to be 127 so the computed estimate works to 2^(127 - 127) = 1
         const __mi sign_exp_set = S::_mm_set1_epi32(0x3F800000);
         result = S::_mm_xor_si(result, sign_exp_set);

         // now we want to -1 to transform [1, 2) to [0, 1), reinterpreting our int bits as float bits
         __m one = S::_mm_set1_ps(1.0f);
         __m floats = S::_mm_sub_ps(_mm_castsi_ps(result), one);

         return std::bit_cast<std::array<float, BATCH_SIZE>>(floats);
      }
      else if constexpr(I == InstructionSet::AVX512) {

      }
   
   }

   [[nodiscard]]
   std::array<uint32_t, BATCH_SIZE> getBatchInts() {
      return std::bit_cast<std::array<uint32_t, BATCH_SIZE>>(advance());   
   }

   /**
    * @brief generate a batch of 8 floats
    * This is a wrapper around getFloatBatch() allowing a more explicit name at the callsite.
    * If you are not generating 8 floats in a batch this will compile error to warn you.
    * You are likely not using AVX256 which is the problem if you get a warning.
    * @return std::array<float, 8> container holding the random numbers
    */
   [[nodiscard]]
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
    [[nodiscard]]
   std::array<float, 16> get_sixteen_floats() requires (BATCH_SIZE == 16) {
      return getBatchFloats();
   }

private:
   // SOA style layout, BATCH_SIZE number of simultaneous RNGs, state split across 2 arrays
   alignas(REGISTER_BYTE_SIZE) std::array<uint32_t, BATCH_SIZE> a_states;
   alignas(REGISTER_BYTE_SIZE) std::array<uint32_t, BATCH_SIZE> b_states;

   /**
    * @brief Advances each nested RNG one state forward
    * 
    * @return __m(register_size)i of result register
    */
   [[nodiscard]]
   auto advance() {
      if constexpr (I == InstructionSet::AVX256) {
         __m256i avx_a = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_states.data()));
         __m256i avx_b = _mm256_load_si256(reinterpret_cast<const __m256i*>(b_states.data()));
         __m256i mult = _mm256_set1_epi32(0x9E3779BB);
         __m256i result = _mm256_mullo_epi32(avx_a, mult);

         avx_b = _mm256_xor_si256(avx_a, avx_b);
         avx_a = rotl(avx_a, 26);
         avx_a = _mm256_xor_si256(avx_a, avx_b);
         avx_a = _mm256_xor_si256(avx_a, _mm256_slli_epi32(avx_b, 9));
         avx_b = rotl(avx_b, 13);
         
         _mm256_store_si256(reinterpret_cast<__m256i*>(a_states.data()), avx_a);
         _mm256_store_si256(reinterpret_cast<__m256i*>(b_states.data()), avx_b);

         return result;
      }
   }

   [[nodiscard]]
   __m256i rotl(__m256i x, const int k) {
      return _mm256_or_si256(
         _mm256_slli_epi32(x, k), 
         _mm256_srli_epi32(x, 32 - k)
      );
   }
};