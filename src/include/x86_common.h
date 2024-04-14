#pragma once
#include <x86intrin.h>

#define SIMD_INLINE inline __attribute__ ((always_inline))

template<int part> 
SIMD_INLINE __m128i unpack_u8_sse(__m128i a, __m128i b = _mm_setzero_si128())
{
  if(part == 0)
  {
    return _mm_unpacklo_epi8(a, b);
  }
  else
  {
    return _mm_unpackhi_epi8(a, b);
  }
}

template<int part> 
SIMD_INLINE __m128i unpack_u16_sse(__m128i a, __m128i b = _mm_setzero_si128())
{
  if(part == 0)
  {
    return _mm_unpacklo_epi16(a, b);
  }
  else
  {
    return _mm_unpackhi_epi16(a, b);
  }
}

template<int part> 
SIMD_INLINE __m128i unpack_u32_sse(__m128i a, __m128i b = _mm_setzero_si128())
{
  if(part == 0)
  {
    return _mm_unpacklo_epi32(a, b);
  }
  else
  {
    return _mm_unpackhi_epi32(a, b);
  }
}

template<int part> 
SIMD_INLINE __m256i unpack_u8_avx2(__m256i a, __m256i b = _mm256_setzero_si256())
{
  if(part == 0)
  {
    return _mm256_unpacklo_epi8(a, b);
  }
  else
  {
    return _mm256_unpackhi_epi8(a, b);
  }
}

template <int part>
SIMD_INLINE __m256i unpack_u16_avx2(__m256i a, __m256i b = _mm256_setzero_si256())
{
  if (part == 0)
  {
    return _mm256_unpacklo_epi16(a, b);
  }
  else
  {
    return _mm256_unpackhi_epi16(a, b);
  }
}

template <int part>
SIMD_INLINE __m256i unpack_u32_avx2(__m256i a, __m256i b = _mm256_setzero_si256())
{
  if (part == 0)
  {
    return _mm256_unpacklo_epi32(a, b);
  }
  else
  {
    return _mm256_unpackhi_epi32(a, b);
  }
}