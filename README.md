# Xoroshiro64StarSIMD

Header-only C++20 SIMD implementation of **xoroshiro64\*** (a fast small-state PRNG by David Blackman and Sebastiano Vigna), specialized for generating batches of 32-bit values efficiently.

This repo provides:
- `XoroshiroRNG`, a vectorized generator that advances multiple xoroshiro64\* streams in parallel for:
    - Batch APIs returning `std::array<uint32_t, BATCH_SIZE>` or `std::array<float, BATCH_SIZE>`.
    - Fill APIs for writing aligned output directly to memory.
- `SequentialXoroshiroRNG`, a wrapper around the vectorized generator
    - Single element float/uint32_t generation through a refilling buffer

## Bench Results

```
scalar(xor)       : 0.429835 s  (1395.89 M u32/s)  // xoring bench
sequential(xor)   : 0.266961 s  (2247.52 M u32/s)  // ~1.61x faster
simd(xor)         : 0.059175 s  (10139.46 M u32/s) // ~7.26x faster
scalar(fill)      : 0.127893 s  (1563.80 M u32/s)  // array bench
simd(fill)        : 0.019482 s  (10265.81 M u32/s) // ~6.56x faster
```
Very large speed improvements especially with the batch api, more details on measurement are at the bottom.

## Notes

- The xoroshiro-family generators like this one are **not** cryptographically secure.
- My implementation modifies the advance function to mildly mix across pairs of streams at no performance cost.
    - After this it performed slightly better on TestU01 passing the normal Crush.
    - Disable it if concerned by setting the default structural type on `advance()` to `false`.
- The fill APIs use asserts to check memory is properly aligned, so this won't be checked on release builds.
- Using `alignas` on a vector doesn't require its heap allocated resource to be aligned, so be wary of that.

## Requirements / Suggestions

- x86/x86_64 CPU and a compiler that supports Intel/AMD SIMD intrinsics.
- Compile with at least AVX enabled (`-mavx`, `-mavx2`, `-mavx512f`, or `-march=native`).

    - Note: compiling without AVX support will work but a warning will be produced as all speed benefits are lost.
- C++20 (minimal modification needed to make it compatible with older standards though)

## Instruction set selection

The header auto-selects the best available at compile time:

- AVX-512 (`__AVX512F__`) -> 512-bit registers -> `BATCH_SIZE = 16` floats/uint32 values
- AVX2 (`__AVX2__`) -> 256-bit registers -> `BATCH_SIZE = 8`
- AVX (`__AVX__`) -> 128-bit registers -> `BATCH_SIZE = 4`

Else it runs just a standard non vectorised version with `BATCH_SIZE = 1`, providing a warning at compile time.

## Usage

Include the header and pull batches:

```cpp
#include "xoroshiro64star.hpp"

int main() {
    XoroshiroRNG batch_rng(123u);

    auto uints = batch_rng.get_batch_uint32();   // std::array<uint32_t, XoroshiroRNG::BATCH_SIZE>
    auto floats = batch_rng.get_batch_floats();   // std::array<float,    XoroshiroRNG::BATCH_SIZE> in [0, 1)

    SequentialXoroshiroRNG sqn_rng(123u);
    uint32_t u = sqn_rng.get_uint32();
    float f = sqn_rng.get_float();
}
```

### Filling aligned buffers

`fill_aligned_uint32()` and `fill_aligned_float()` require the destination pointer to be aligned to `XoroshiroRNG::REGISTER_BYTE_SIZE`.

```cpp
#include "xoroshiro64star.hpp"
#include <array>

int main() {
    XoroshiroRNG rng;

    alignas(XoroshiroRNG::REGISTER_BYTE_SIZE)
    std::array<uint32_t, 1024> out{};

    rng.fill_aligned_uint32(out.data(), out.size());
}
```

## Benchmarking Details

`benchmark.cpp` compares a scalar xoroshiro64\* loop against the SIMD batch API.

The benchmark noted above was compiled on a AMD Ryzen 7 260 (zen 4 architecture). It only has 256 bit SIMD registers but supports the AVX512 instruction set. It does this by double pumping the 256 bit vector register, so it 2 cycles for an operation but its generally the same speed or faster than 2 separate 256 bit instructions.

On a CPU with true 512 bit vector registers the simd version will have a greater advantage, and for a older CPU not support AVX2 the benefit would be less significant.

Benchmark was compiled with:

```bash
g++ -O3 -std=c++20 -mavx512f -flto -funroll-loops -fno-exceptions -fno-rtti \
    -march=native -fomit-frame-pointer benchmark.cpp -o benchmark && ./benchmark
```

Build and run example:

```bash
g++ -O3 -std=c++20 -march=native -flto -funroll-loops \
    -fno-exceptions -fno-rtti -fomit-frame-pointer \
    benchmark.cpp -o benchmark

./benchmark
```

To target AVX-512, compile with something like `-mavx512f` (or just `-march=native` to just inform the compiler of your hardware).

## Credits

- xoroshiro64\* algorithm: David Blackman and Sebastiano Vigna.
- SIMD implementation in this repo: see the header comment in `xoroshiro64star.hpp`.
