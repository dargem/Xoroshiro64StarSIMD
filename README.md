# Xoroshiro64StarSIMD

Header-only C++20 SIMD implementation of **xoroshiro64\*** (a fast small-state PRNG by David Blackman and Sebastiano Vigna), specialized for generating batches of 32-bit values efficiently.

This repo provides:
- `XoroshiroRNG`, a vectorized generator that advances multiple xoroshiro64\* streams in parallel for:
    - Batch APIs generating float/uint32_t/int32_t arrays, e.g. `std::array<uint32_t, BATCH_SIZE>`.
    - Fill APIs for writing output directly to SIMD (Vector Register) aligned or 4 byte aligned memory.
- `SequentialXoroshiroRNG`, a wrapper around the vectorized generator
    - Single element float/uint32_t/int32_t generation through a refilling buffer

## Bench Results

```
mersenne(xor)     : 1.335189 s  (449.37 M u32/s)   // baseline
scalar(xor)       : 0.407096 s  (1473.85 M u32/s)  // ~3.28x faster
sequential(xor)   : 0.263601 s  (2276.17 M u32/s)  // ~5.07x faster
simd(xor)         : 0.058874 s  (10191.23 M u32/s) // ~22.7x faster

mersenne(fill)    : 0.461094 s  (433.75 M u32/s)   // baseline for aligned fills
scalar(fill)      : 0.126347 s  (1582.94 M u32/s)  // ~3.65x faster
simd(fill)        : 0.019802 s  (10099.97 M u32/s) // ~23.3x faster
```
Very large speed improvements especially with the batch api, more details on measurement are at the bottom.

## Notes

- The xoroshiro-family generators like this one are **not** cryptographically secure.
- My implementation modifies the advance function to mildly mix across pairs of streams at no performance cost.
    - After this it performed slightly better on TestU01 passing the normal Crush.
    - Disable it if concerned by setting the default structural type on `advance()` to `false`.
- The fill APIs use asserts to check memory is properly aligned, so this won't be checked on release builds.
- Using `alignas` on a vector doesn't require its heap allocated resource to be SIMD aligned, so be wary of that.

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

Include the header and pull raw batches or scalar values:

```cpp
#include "xoroshiro64star.hpp"

int main() {
    XoroshiroRNG batch_rng(123u);

    auto uints = batch_rng.get_batch_uint32();   // std::array<uint32_t, XoroshiroRNG::BATCH_SIZE>
    auto floats = batch_rng.get_batch_floats();   // std::array<float,    XoroshiroRNG::BATCH_SIZE> in [0, 1)

    // The SequentialXoroshiroRNG can be templated with capabilities, which is the types it can generate. 
    // Since it uses a different buffer for int/float if you don't want to pay the space cost of the floats
    SequentialXoroshiroRNG<int32_t> int_generator;
    int32_t sink = int_generator.get_int32();

    // Fails to compile, our generator does not support get float
    // float f = int_generator.get_float(); 

    SequentialXoroshiroRNG<int32_t, float> num_generator;
    sink = num_generator.get_int32();
    float f = num_generator.get_float(); // Works fine since they are both capabilities

    std::cout << "sizeof int_gen: " << sizeof(int_generator) << '\n';
    std::cout << "sizeof num_gen: " << sizeof(num_generator) << std::endl;
}
```

### Filling aligned and partially aligned buffers

`fill_aligned_uint32()` and `fill_aligned_float()` require the destination pointer to be aligned to `XoroshiroRNG::REGISTER_BYTE_SIZE`.
`fill_partial_aligned_uint32()` and `fill_partial_aligned_float()` require the destination pointer to be aligned to `sizeof(data_type)` aka 4 bytes.

```cpp
#include "xoroshiro64star.hpp"
#include <array>
#include <vector>

int main() {
    XoroshiroRNG rng;

    alignas(XoroshiroRNG::REGISTER_BYTE_SIZE)
    std::array<uint32_t, 1024> out{};

    rng.fill_aligned_uint32(out.data(), out.size());

    // alignas(XoroshiroRNG::REGISTER_BYTE_SIZE) does not require 
    // its heap allocated resource to be aligned  using a fill_aligned 
    // method is undefined behaviour and can cause out of bounds access
    std::vector<uint32_t> out2{};
    out2.resize(1024);

    // There is a one time cost on each call to fill in memory until it reaches a SIMD address.
    // Then it can do an aligned fill so overhead does not scale with size of data.
    // Custom allocators can be used to allocate aligned memory for vectors if this is a concern.
    rng.fill_partial_aligned_uint32(out2.data(), out2.size());
}
```

## Benchmarking Details

`benchmark.cpp` compares a scalar xoroshiro64\* loop against the SIMD batch API and its buffered/sequential version.

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

To target AVX-512, compile with something like `-mavx512f` (or just `-march=native` to just inform the compiler of your hardware). Compiling without it leaves no performance benefits but it works out to be the same speed as the original scalar impl.

```
xoroshiro64star.hpp:276:2: warning: #warning Being ran on a CPU that does not support any AVX instruction sets. Or it may have been compiled without targeting your architecture (flags march native or specify mavx). Expect no benefits from vectorization. [-Wcpp]
  276 | #warning Being ran on a CPU that does not support any AVX instruction sets. \
      |  ^~~~~~~
Benchmark generating
mersenne(xor)     : 1.593589 s  (376.51 M u32/s)
scalar(xor)       : 0.405424 s  (1479.93 M u32/s)
sequential(xor)   : 0.360740 s  (1663.25 M u32/s)
simd(xor)         : 0.405872 s  (1478.30 M u32/s)

mersenne(fill)    : 0.549555 s  (363.93 M u32/s)
scalar(fill)      : 0.119372 s  (1675.44 M u32/s)
simd(fill)        : 0.119353 s  (1675.70 M u32/s)
```

## Credits

- xoroshiro64\* algorithm: David Blackman and Sebastiano Vigna.
- SIMD implementation in this repo: see the header comment in `xoroshiro64star.hpp`.

The original impl:
https://prng.di.unimi.it/xoroshiro64star.c