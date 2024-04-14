#include "distortion.h"
#include "x86_common.h"
#include <cassert>

uint64_t ssd8u_avx2(
    const uint8_t *o1,
    const uint8_t *o2,
    int O,
    int w,
    int h)
{
  assert(w * h <= 65536);

  constexpr int A = sizeof(__m256i);
  const int wA = w & ~(A - 1);

  __m256i acc32 = _mm256_setzero_si256();
  uint64_t ssd = 0;

  for (int y = 0; y < h; ++y)
  {
    int x;
    for (x = 0; x < wA; x += A)
    {
      __m256i s0 = _mm256_loadu_si256((__m256i *)&o1[y * O + x]);
      __m256i s1 = _mm256_loadu_si256((__m256i *)&o2[y * O + x]);

      __m256i us0 = unpack_u8_avx2<0>(s0);
      __m256i us1 = unpack_u8_avx2<0>(s1);
      us0 = _mm256_sub_epi16(us0, us1);
      us0 = _mm256_madd_epi16(us0, us0);
      acc32 = _mm256_add_epi32(acc32, us0);

      us0 = unpack_u8_avx2<1>(s0);
      us1 = unpack_u8_avx2<1>(s1);
      us0 = _mm256_sub_epi16(us0, us1);
      us0 = _mm256_madd_epi16(us0, us0);
      acc32 = _mm256_add_epi32(acc32, us0);
    }
    for (; x < w; ++x)
    {
      int d = (int)o1[y * O + x] - (int)o2[y * O + x];
      ssd += (d * d);
    }
  }

  acc32 = _mm256_hadd_epi32(acc32, acc32);
  acc32 = _mm256_hadd_epi32(acc32, acc32);

  ssd += (uint64_t)_mm256_extract_epi32(acc32, 0);
  ssd += (uint64_t)_mm256_extract_epi32(acc32, 4);

  return ssd;
}

uint64_t ssd10u_avx2(
    const uint8_t *src1,
    const uint8_t *src2,
    int src_stride,
    int w,
    int h)
{
  constexpr int A = sizeof(__m256i) / 2;
  const int wA = w & ~(A - 1);
  const uint16_t *o1 = (const uint16_t *)src1;
  const uint16_t *o2 = (const uint16_t *)src2;
  const int O = src_stride / 2;

  __m256i acc64 = _mm256_setzero_si256();
  uint64_t ssd = 0;

  for (int y = 0; y < h; ++y)
  {
    int x;
    for (x = 0; x < wA; x += A)
    {
      __m256i s0 = _mm256_loadu_si256((__m256i *)&o1[y * O + x]);
      __m256i s1 = _mm256_loadu_si256((__m256i *)&o2[y * O + x]);

      s0 = _mm256_sub_epi16(s0, s1);
      s0 = _mm256_madd_epi16(s0, s0);
      s0 = _mm256_hadd_epi32(s0, s0);
      acc64 = _mm256_add_epi64(acc64, unpack_u32_avx2<0>(s0));
    }
    for (; x < w; ++x)
    {
      int d = (int)o1[y * O + x] - (int)o2[y * O + x];
      ssd += (d * d);
    }
  }

  ssd += (uint64_t)_mm256_extract_epi64(acc64, 0);
  ssd += (uint64_t)_mm256_extract_epi64(acc64, 1);
  ssd += (uint64_t)_mm256_extract_epi64(acc64, 2);
  ssd += (uint64_t)_mm256_extract_epi64(acc64, 3);

  return ssd;
}