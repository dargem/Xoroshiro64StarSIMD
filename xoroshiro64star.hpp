/* Written in 2016 by David Blackman and Sebastiano Vigna (vigna@acm.org).
 * SIMD implementation produced by me. https://github.com/dargem/Xoroshiro64StarSIMD.

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

#ifndef XOROSHIRO_64_STAR_HPP
#define XOROSHIRO_64_STAR_HPP

#include <cstdint>
#include <cstddef>
#include <array>
#include <cassert>
#include <immintrin.h>
#include <cstring>
#include <bit>

enum class InstructionSet {
   NONE,
   AVX128, // AKA AVX
   AVX256, // AKA AVX2
   AVX512
};

template <InstructionSet I>
struct InstructionSetTraits;

template<>
struct InstructionSetTraits<InstructionSet::NONE> {
   // Here we're using just normal registers which are 64 bits yes, but 
   // we're just using this for 4 byte data so we treat them as 32 bit for this. 
   // SWAR could be used but thats outside this projects scope.
   static constexpr size_t bits = 32;
   static constexpr size_t bytes = bits / 8;

   using __mi = uint32_t;
   using __m = float;

   // Integer ops
   template <int bit_shift>
   static __mi rol_epi32(__mi a) { 
      return (a << bit_shift) | (a >> (32 - bit_shift));
   }

   template <int bit_shift>
   static __mi rol_epi64(__mi a) { 
      return rol_epi32<bit_shift>(a); // We just fall back to 32 bit
   }

   static __mi srli_epi32(__mi a, int bits) { return a >> bits; }
   static __mi slli_epi32(__mi a, int bits) { return a << bits; }
   static __mi set1_epi32(int val) { return val; }
   static __mi mullo_epi32(__mi a, __mi b) { return a * b; }

   // Scalar fallback, treat "epi64" ops as operating on the scalar value
   static __mi slli_epi64(__mi a, int bits) { return slli_epi32(a, bits); }

   // Bit ops
   static __mi xor_si(__mi a, __mi b) { return a ^ b; }
   static __mi or_si(__mi a, __mi b) { return a | b; }
   static void store_si(__mi* mem_addr, __mi source) { *mem_addr = source; }
   static __mi load_si(__mi const* mem_addr) { return *mem_addr; }

   // Float ops
   static __m sub_ps(__m a, __m b) { return a - b; }
   static __m set1_ps(float val) { return val; }
   static __m castsi_ps(__mi a) { return std::bit_cast<__m>(a); }
   static void store_ps(float* mem_addr, __m source) { *mem_addr = source; }
};

#ifdef __AVX__
template<>
struct InstructionSetTraits<InstructionSet::AVX128> {
   static constexpr size_t bits = 128;
   static constexpr size_t bytes = bits / 8;

   using __mi = __m128i;
   using __m = __m128;

   // Integer ops
   template <int bit_shift>
   static __mi rol_epi32(__mi a) { 
      #if defined (__AVX512F__)
         // this 128 bit register instruction is actually from the AVX512 instruction set
         return _mm_rol_epi32(a, bit_shift); 
      #else
         // if we do not have the AVX512 instruction set default back to our 128 "emulation"
         return or_si(
            slli_epi32(a, bit_shift), 
            srli_epi32(a, 32 - bit_shift)
         );
      #endif
   }

   template <int bit_shift>
   static __mi rol_epi64(__mi a) { 
      #if defined (__AVX512F__)
         // this 128 bit register instruction is actually from the AVX512 instruction set
         return _mm_rol_epi64(a, bit_shift); 
      #else
         // if we do not have the AVX512 instruction set default back to our 128 "emulation"
         return or_si(
            slli_epi64(a, bit_shift), 
            srli_epi64(a, 64 - bit_shift)
         );
      #endif
   }

   static __mi srli_epi32(__mi a, int bits) { return _mm_srli_epi32(a, bits); }
   static __mi slli_epi32(__mi a, int bits) { return _mm_slli_epi32(a, bits); }
   static __mi set1_epi32(int val) { return _mm_set1_epi32(val); }
   static __mi mullo_epi32(__mi a, __mi b) { return _mm_mullo_epi32(a, b); }
   static __mi srli_epi64(__mi a, int bits) { return _mm_srli_epi64(a, bits); }
   static __mi slli_epi64(__mi a, int bits) { return _mm_slli_epi64(a, bits); }
   static __mi add_epi64(__mi a, __mi b) { return _mm_add_epi64(a, b); }
   static __mi set1_epi64(uint64_t a) { return _mm_set1_epi64x(a); }

   // Bit ops
   static __mi xor_si(__mi a, __mi b) { return _mm_xor_si128(a, b); }
   static __mi or_si(__mi a, __mi b) { return _mm_or_si128(a, b); }
   static void store_si(__mi* mem_addr, __mi source) { _mm_store_si128(mem_addr, source); }
   static __mi load_si(__mi const* mem_addr) { return _mm_load_si128(mem_addr); }

   // Float ops
   static __m sub_ps(__m a, __m b) { return _mm_sub_ps(a, b); }
   static __m set1_ps(float val) { return _mm_set1_ps(val); }
   static __m castsi_ps(__mi a) { return _mm_castsi128_ps(a); }
   static void store_ps(float* mem_addr, __m source) { _mm_store_ps(mem_addr, source); }
   // static void store_si(__mi* mem_addr, __mi source) { _mm_store_si128(mem_addr, source); }
};
#endif

#ifdef __AVX2__
template<>
struct InstructionSetTraits<InstructionSet::AVX256> {
   static constexpr size_t bits = 256;
   static constexpr size_t bytes = bits / 8;

   using __mi = __m256i;
   using __m = __m256;

   // Integer ops
   template <int bit_shift>
   static __mi rol_epi32(__mi a) { 
      #if defined (__AVX512F__)
         // this 256 bit register instruction is actually from the AVX512 instruction set
         return _mm256_rol_epi32(a, bit_shift); 
      #else
         // if we do not have the AVX512 instruction set default back to our 256 "emulation"
         return or_si(
            slli_epi32(a, bit_shift), 
            srli_epi32(a, 32 - bit_shift)
         );
      #endif
   }

   template <int bit_shift>
   static __mi rol_epi64(__mi a) { 
      #if defined (__AVX512F__)
         // this 256 bit register instruction is actually from the AVX512 instruction set
         return _mm256_rol_epi64(a, bit_shift); 
      #else
         // if we do not have the AVX512 instruction set default back to our 256 "emulation"
         return or_si(
            slli_epi64(a, bit_shift), 
            srli_epi64(a, 64 - bit_shift)
         );
      #endif
   }

   static __mi srli_epi32(__mi a, int bits) { return _mm256_srli_epi32(a, bits); }
   static __mi slli_epi32(__mi a, int bits) { return _mm256_slli_epi32(a, bits); }
   static __mi set1_epi32(int val) { return _mm256_set1_epi32(val); }
   static __mi mullo_epi32(__mi a, __mi b) { return _mm256_mullo_epi32(a, b); }
   static __mi srli_epi64(__mi a, int bits) { return _mm256_srli_epi64(a, bits); }
   static __mi slli_epi64(__mi a, int bits) { return _mm256_slli_epi64(a, bits); }
   static __mi add_epi64(__mi a, __mi b) { return _mm256_add_epi64(a, b); }
   static __mi set1_epi64(uint64_t a) { return _mm256_set1_epi64x(a); }

   // Bit ops
   static __mi xor_si(__mi a, __mi b) { return _mm256_xor_si256(a, b); }
   static __mi or_si(__mi a, __mi b) { return _mm256_or_si256(a, b); }
   static void store_si(__mi* mem_addr, __mi source) { _mm256_store_si256(mem_addr, source); }
   static __mi load_si(__mi const* mem_addr) { return _mm256_load_si256(mem_addr); }

   // Float ops
   static __m sub_ps(__m a, __m b) { return _mm256_sub_ps(a, b); }
   static __m set1_ps(float val) { return _mm256_set1_ps(val); }
   static __m castsi_ps(__mi a) { return _mm256_castsi256_ps(a); }
   static void store_ps(float* mem_addr, __m source) { _mm256_store_ps(mem_addr, source); }
};
#endif


#ifdef __AVX512F__
template<>
struct InstructionSetTraits<InstructionSet::AVX512> {
   static constexpr size_t bits = 512;
   static constexpr size_t bytes = bits / 8;

   using __mi = __m512i;
   using __m = __m512;

   // Integer ops
   template <int bit_shift> // needs a compile time constant
   static __mi rol_epi32(__mi a) { return _mm512_rol_epi32(a, bit_shift); }

   template <int bit_shift> // needs a compile time constant
   static __mi rol_epi64(__mi a) { return _mm512_rol_epi64(a, bit_shift); }

   static __mi srli_epi32(__mi a, int bits) { return _mm512_srli_epi32(a, bits); }
   static __mi slli_epi32(__mi a, int bits) { return _mm512_slli_epi32(a, bits); }
   static __mi set1_epi32(int val) { return _mm512_set1_epi32(val); }
   static __mi mullo_epi32(__mi a, __mi b) { return _mm512_mullo_epi32(a, b); }
   static __mi srli_epi64(__mi a, int bits) { return _mm512_srli_epi64(a, bits); }
   static __mi slli_epi64(__mi a, int bits) { return _mm512_slli_epi64(a, bits); }
   static __mi add_epi64(__mi a, __mi b) { return _mm512_add_epi64(a, b); }
   static __mi set1_epi64(uint64_t a) { return _mm512_set1_epi64(a); }

   // Bit ops
   static __mi xor_si(__mi a, __mi b) { return _mm512_xor_si512(a, b); }
   static __mi or_si(__mi a, __mi b) { return _mm512_or_si512(a, b); }
   static void store_si(__mi* mem_addr, __mi source) { _mm512_store_si512(mem_addr, source); }
   static __mi load_si(const __mi* mem_addr) { return _mm512_load_si512(mem_addr); } 

   // Float ops
   static __m sub_ps(__m a, __m b) { return _mm512_sub_ps(a, b); }
   static __m set1_ps(float val) { return _mm512_set1_ps(val); }
   static __m castsi_ps(__mi a) { return _mm512_castsi512_ps(a); }
   static void store_ps(float* mem_addr, __m source) { _mm512_store_ps(mem_addr, source); }
};
#endif

static inline uint64_t splitmix64_next(uint64_t& x) {
   // http://xorshift.di.unimi.it/splitmix64.c
   uint64_t z = (x += 0x9e3779b97f4a7c15);
   z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
   z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
   return z ^ (z >> 31);
}

#ifdef __AVX512F__
#define SIMD_INSTRUCTION_SET InstructionSet::AVX512   
#elif defined (__AVX2__)
#define SIMD_INSTRUCTION_SET InstructionSet::AVX256
#elif defined (__AVX__)
#define SIMD_INSTRUCTION_SET InstructionSet::AVX128
#else
#warning Being ran on a CPU that does not support any AVX instruction sets. \
Or it may have been compiled without targeting your architecture (flags march native or specify mavx). \
Expect no benefits from vectorization.
#define SIMD_INSTRUCTION_SET InstructionSet::NONE
#endif


class alignas(InstructionSetTraits<SIMD_INSTRUCTION_SET>::bytes) XoroshiroRNG {
public:
   using _mm = InstructionSetTraits<SIMD_INSTRUCTION_SET>;
   using __m = _mm::__m;   // float
   using __mi = _mm::__mi; // int

   constexpr static size_t REGISTER_BYTE_SIZE = _mm::bytes;
   constexpr static size_t ELEMENT_SIZE = sizeof(float);
   constexpr static size_t BATCH_SIZE = REGISTER_BYTE_SIZE / ELEMENT_SIZE;
   static constexpr InstructionSet INSTRUCTION_SET = SIMD_INSTRUCTION_SET;
   
   XoroshiroRNG(uint32_t seed = 0xcafef00dU) {

      uint64_t sm = 0xfeedfacecafebeefULL ^ uint64_t(seed);

      for (size_t i{}; i < BATCH_SIZE; ++i) {
         a_states[i] = splitmix64_next(sm);
         b_states[i] = splitmix64_next(sm);
      }

   }

   /**
    * @brief Get a batch of floats in an array container
    * The number of floats is the SIMD register size in bits / 32. 
    * AVX512 uses 512 bit registers, so batch size is 512 / 32 = 16.
    * @return std::array<float, BATCH_SIZE> 
    */
   [[nodiscard]]
   std::array<float, BATCH_SIZE> get_batch_floats() {

      __mi result = advance();

      return std::bit_cast<std::array<float, BATCH_SIZE>>(float_convert(result));
   }

   /**
    * @brief Get a batch of floats in raw simd register
    * The number of float is the SIMD register size in bits / 32. 
    * AVX512 uses 512 bit registers, so batch size is 512 / 32 = 16.
    * @return __m, which is a SIMD float register
    */
   [[nodiscard]]
   __m get_batch_floats_simd() {
      return float_convert(advance());   
   }

   /**
    * @brief Get a batch of uint32_t's in an array container
    * The number of ints is the SIMD register size in bits / 32. 
    * AVX512 uses 512 bit registers, so batch size is 512 / 32 = 16.
    * @return std::array<int, BATCH_SIZE> 
    */
   [[nodiscard]]
   std::array<uint32_t, BATCH_SIZE> get_batch_uint32() {
      return std::bit_cast<std::array<uint32_t, BATCH_SIZE>>(advance());   
   }

   /**
    * @brief Get a batch of uint32_t's in raw simd register
    * The number of ints is the SIMD register size in bits / 32. 
    * AVX512 uses 512 bit registers, so batch size is 512 / 32 = 16.
    * @return __mi, which is a SIMD integer register
    */
   [[nodiscard]]
   __mi get_batch_uint32_simd() {
      return advance();   
   }

   /**
    * @brief Fill an array with uint32_t's
    * 
    * @param dst Start address, must be aligned to your SIMD register size
    * @param num_elements Number of uint32_t's to fill it with 
    */
   void fill_aligned_uint32(uint32_t* dst, size_t num_elements) {
      // This is necessary to do a faster load instruction since we don't need to worry about alignment
      assert(reinterpret_cast<uintptr_t>(dst) % REGISTER_BYTE_SIZE == 0 && "destination must be aligned to REGISTER_BYTE_SIZE");

      uint32_t* end = dst + num_elements;

      // We want the aligned end to fill in batches until we reach that
      size_t remainder_bytes = reinterpret_cast<uintptr_t>(end) % REGISTER_BYTE_SIZE;
      uint32_t* aligned_end = end - remainder_bytes / sizeof(uint32_t);

      // Fill all elements up to the SIMD registers boundary
      for (; dst < aligned_end; dst += BATCH_SIZE) {
         __mi result = advance();
         _mm::store_si(reinterpret_cast<__mi*>(dst), result);
      }

      if (remainder_bytes != 0) {
         // Now we have to fill the remainders if it exists
         __mi buffer = advance();
         std::memcpy(dst, &buffer, remainder_bytes);
      }
   }

   /**
    * @brief Fill an array with uint32_t's
    * Require partial alignment. Must be 4 byte aligned but doesn't require it to be SIMD register size aligned.
    * @param dst Start address
    * @param num_elements Number of uint32_t's to fill it with 
    */
   void fill_partial_aligned_uint32(uint32_t* dst, size_t num_elements) {

      // Doesn't need to be SIMD aligned but still needs to be 4 byte aligned
      assert(reinterpret_cast<uintptr_t>(dst) % sizeof(uint32_t) == 0 && "destination must be 4 byte aligned");

      // We want to fill until the next SIMD register aligned point, then we can just use an aligned fill
      size_t bytes_needed = (REGISTER_BYTE_SIZE - reinterpret_cast<uintptr_t>(dst) % REGISTER_BYTE_SIZE) % REGISTER_BYTE_SIZE;
      size_t elements_needed_to_align = bytes_needed / sizeof(uint32_t);

      if (num_elements <= elements_needed_to_align) {
         // We only need to fill num_elements here then we can early return
         __mi buffer = advance();
         std::memcpy(dst, &buffer, num_elements * sizeof(uint32_t));
         return;
      }

      if (bytes_needed != 0) {
         __mi buffer = advance();
         std::memcpy(dst, &buffer, bytes_needed);
         dst += elements_needed_to_align; // We have now SIMD register aligned it
      }

      // dst is now aligned and we can delegate to fill_aligned
      fill_aligned_uint32(dst, num_elements - elements_needed_to_align);
   }
   
   /**
    * @brief Fill an array with int32_t's
    * 
    * @param dst Start address, must be aligned to your SIMD register size
    * @param num_elements Number of int32_t's to fill it with 
    */
   void fill_aligned_int32(int32_t* dst, size_t num_elements) {
      // This is necessary to do a faster load instruction since we don't need to worry about alignment
      assert(reinterpret_cast<uintptr_t>(dst) % REGISTER_BYTE_SIZE == 0 && "destination must be aligned to REGISTER_BYTE_SIZE");

      int32_t* end = dst + num_elements;

      // We want the aligned end to fill in batches until we reach that
      size_t remainder_bytes = reinterpret_cast<uintptr_t>(end) % REGISTER_BYTE_SIZE;
      int32_t* aligned_end = end - remainder_bytes / sizeof(int32_t);

      // Fill all elements up to the SIMD registers boundary
      for (; dst < aligned_end; dst += BATCH_SIZE) {
         __mi result = advance();
         _mm::store_si(reinterpret_cast<__mi*>(dst), result);
      }

      if (remainder_bytes != 0) {
         // Now we have to fill the remainders if it exists
         __mi buffer = advance();
         std::memcpy(dst, &buffer, remainder_bytes);
      }
   }

   /**
    * @brief Fill an array with int32_t's
    * Require partial alignment. Must be 4 byte aligned but doesn't require it to be SIMD register size aligned.
    * @param dst Start address
    * @param num_elements Number of int32_t's to fill it with 
    */
   void fill_partial_aligned_int32(int32_t* dst, size_t num_elements) {

      // Doesn't need to be SIMD aligned but still needs to be 4 byte aligned
      assert(reinterpret_cast<uintptr_t>(dst) % sizeof(int32_t) == 0 && "destination must be 4 byte aligned");

      // We want to fill until the next SIMD register aligned point, then we can just use an aligned fill
      size_t bytes_needed = (REGISTER_BYTE_SIZE - reinterpret_cast<uintptr_t>(dst) % REGISTER_BYTE_SIZE) % REGISTER_BYTE_SIZE;
      size_t elements_needed_to_align = bytes_needed / sizeof(int32_t);

      if (num_elements <= elements_needed_to_align) {
         // We only need to fill num_elements here then we can early return
         __mi buffer = advance();
         std::memcpy(dst, &buffer, num_elements * sizeof(int32_t));
         return;
      }

      if (bytes_needed != 0) {
         __mi buffer = advance();
         std::memcpy(dst, &buffer, bytes_needed);
         dst += elements_needed_to_align; // We have now SIMD register aligned it
      }

      // dst is now aligned and we can delegate to fill_aligned
      fill_aligned_int32(dst, num_elements - elements_needed_to_align);
   }

   /**
    * @brief Fill an array with floats
    * 
    * @param dst Start address, must be aligned to your SIMD register size
    * @param num_elements Number of floats to fill it with 
    */
   void fill_aligned_float(float* dst, size_t num_elements) {
      // This is necessary to do a faster load instruction since we don't need to worry about alignment
      assert(reinterpret_cast<uintptr_t>(dst) % REGISTER_BYTE_SIZE == 0 && "destination must be aligned to REGISTER_BYTE_SIZE");

      float* end = dst + num_elements;

      // We want the aligned end to fill in batches until we reach that
      size_t remainder_bytes = reinterpret_cast<uintptr_t>(end) % REGISTER_BYTE_SIZE;
      float* aligned_end = end - remainder_bytes / sizeof(float);

      // Fill all elements up to the SIMD registers boundary
      for (; dst < aligned_end; dst += BATCH_SIZE) {
         __mi result = advance();
         __m floats = float_convert(result);

         _mm::store_ps(dst, floats);
      }

      if (remainder_bytes != 0) {
         // Now we have to fill the remainders if it exists
         __m buffer = float_convert(advance());
         std::memcpy(dst, &buffer, remainder_bytes);
      }
   }

   /**
    * @brief Fill an array with floats
    * Require partial alignment. Must be 4 byte aligned but it doesn't require it to be SIMD register size aligned.
    * @param dst Start address
    * @param num_elements Number of floats to fill it with 
    */
   void fill_partial_aligned_float(float* dst, size_t num_elements) {

      // Doesn't need to be SIMD aligned but still needs to be 4 byte aligned
      assert(reinterpret_cast<uintptr_t>(dst) % sizeof(float) == 0 && "destination must be 4 byte aligned");

      // We want to fill until the next SIMD register aligned point, then we can just use an aligned fill
      size_t bytes_needed = (REGISTER_BYTE_SIZE - reinterpret_cast<uintptr_t>(dst) % REGISTER_BYTE_SIZE) % REGISTER_BYTE_SIZE;
      size_t elements_needed_to_align = bytes_needed / sizeof(float);

      if (num_elements <= elements_needed_to_align) {
         // We only need to fill num_elements here then we can early return
         __m buffer = float_convert(advance());
         std::memcpy(dst, &buffer, num_elements * sizeof(float));
         return;
      }

      if (bytes_needed != 0) {
         __m buffer = float_convert(advance());
         std::memcpy(dst, &buffer, bytes_needed);
         dst += elements_needed_to_align; // We have now SIMD register aligned it
      }

      // dst is now aligned
      fill_aligned_float(dst, num_elements - elements_needed_to_align);
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
   template <bool LaneWeaving = true>
   [[nodiscard]]
   __mi advance() {
      __mi avx_a = _mm::load_si(reinterpret_cast<const __mi*>(a_states.data()));
      __mi avx_b = _mm::load_si(reinterpret_cast<const __mi*>(b_states.data()));
      __mi mult = _mm::set1_epi32(0x9E3779BB);
      __mi result = _mm::mullo_epi32(avx_a, mult);

      if constexpr (LaneWeaving) {
         // Interprets the 32 bit RNG "lanes" as 64 bit ones for additional mixing
         // This results in it being able to fully pass the normal Crush test from TestU01
         avx_b = _mm::xor_si(avx_a, avx_b);
         avx_a = _mm::template rol_epi64<26>(avx_a);
         avx_a = _mm::xor_si(avx_a, avx_b);
         avx_a = _mm::xor_si(avx_a, _mm::slli_epi64(avx_b, 9));
         avx_b = _mm::template rol_epi64<13>(avx_b);
      } else {
         // This is the original version, every lane is independent
         avx_b = _mm::xor_si(avx_a, avx_b);
         avx_a = _mm::template rol_epi32<26>(avx_a);
         avx_a = _mm::xor_si(avx_a, avx_b);
         avx_a = _mm::xor_si(avx_a, _mm::slli_epi32(avx_b, 9));
         avx_b = _mm::template rol_epi32<13>(avx_b);
      }
      
      _mm::store_si(reinterpret_cast<__mi*>(a_states.data()), avx_a);
      _mm::store_si(reinterpret_cast<__mi*>(b_states.data()), avx_b);

      return result;
   }

   [[nodiscard]]
   __mi rotl(__mi x, const int k) {
      return _mm::or_si(
         _mm::slli_epi32(x, k), 
         _mm::srli_epi32(x, 32 - k)
      );
   }

   /**
   * @brief Returns a conversion of given integers into floats
   * 
   * @param data 
   * @return __m 
   */
   __m float_convert(__mi data) {
      
      // Need to convert result into a [0, 1) float
      // bit shift 9 to the right to get rid of sign + 8 bit exponent
      data = _mm::srli_epi32(data, 9);

      // want this to be converted to [0, 1], want a leading 0 so its signed positive, 
      // for a floating point we know number = 2^n * (1 + mantissa), where mantissa = [0, 1)
      // if we have 2^n = 1, then number = 1 + mantissa = [1, 2), 
      // therefore [0, 1) = [1, 2) - 1 = 2^0 * (1 + mantissa)
      // want n to equal 0, but n is found through subtracting exponent field by 127 in a float
      // so we want exponent field to be 127 so the computed estimate works to 2^(127 - 127) = 1
      const __mi sign_exp_set = _mm::set1_epi32(0x3F800000);
      data = _mm::xor_si(data, sign_exp_set);

      // now we want to -1 to transform [1, 2) to [0, 1), reinterpreting our int bits as float bits
      __m one = _mm::set1_ps(1.0f);
      return _mm::sub_ps(_mm::castsi_ps(data), one);
   }
};

class SequentialXoroshiroRNG {
public:
   SequentialXoroshiroRNG(uint32_t seed = 0xcafef00dU) 
      : rng(seed), int_buffer(rng.get_batch_uint32()), float_buffer(rng.get_batch_floats())
   {}

   uint32_t get_uint32() {
      if (int_idx == XoroshiroRNG::BATCH_SIZE) {
         int_idx = 0;
         int_buffer = rng.get_batch_uint32();
      }

      return int_buffer[int_idx++];
   }

   float get_float() {
      if (float_idx == XoroshiroRNG::BATCH_SIZE) {
         float_idx = 0;
         float_buffer = rng.get_batch_floats();
      }

      return float_buffer[float_idx++];
   }

private:
   XoroshiroRNG rng;
   // We need to be aware at a given time whether the buffer is filled with ints or floats
   // So we unfortunately need to use 2 buffers to avoid branching on accesses
   std::array<uint32_t, XoroshiroRNG::BATCH_SIZE> int_buffer;
   std::array<float, XoroshiroRNG::BATCH_SIZE> float_buffer;
   uint8_t int_idx{};
   uint8_t float_idx{};
};

#endif