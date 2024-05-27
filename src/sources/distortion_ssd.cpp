#include "distortion.h"
#include "x86_common.h"
#include <cassert>

template <typename T>
uint64_t ssd_c(
    const uint8_t *ref_ptr,
    const int ref_stride,
    const uint8_t *tar_ptr,
    const int tar_stride,
    const int w,
    const int h)
{
  static_assert(sizeof(T) <= 2);

  const T *o1 = (const T *)ref_ptr;
  const T *o2 = (const T *)tar_ptr;
  const int O1 = ref_stride / sizeof(T);
  const int O2 = tar_stride / sizeof(T);
  uint64_t ssd = 0;

  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      int d = (int)o1[y * O1 + x] - (int)o2[y * O2 + x];
      ssd += (d * d);
    }
  }

  return ssd;
}

uint64_t ssd_8u_sse(
    const uint8_t *o1,
    const int O1,
    const uint8_t *o2,
    const int O2,
    const int w,
    const int h)
{
  assert(w * h <= 65536);

  constexpr int A = sizeof(__m128i);
  const int wA = w & ~(A - 1);

  __m128i acc32 = _mm_setzero_si128();
  uint64_t ssd = 0;

  for (int y = 0; y < h; ++y)
  {
    int x;
    for (x = 0; x < wA; x += A)
    {
      __m128i s0 = _mm_loadu_si128((__m128i *)&o1[y * O1 + x]);
      __m128i s1 = _mm_loadu_si128((__m128i *)&o2[y * O2 + x]);

      __m128i us0 = unpack_u8_sse<0>(s0);
      __m128i us1 = unpack_u8_sse<0>(s1);
      us0 = _mm_sub_epi16(us0, us1);
      us0 = _mm_madd_epi16(us0, us0);
      acc32 = _mm_add_epi32(acc32, us0);

      us0 = unpack_u8_sse<1>(s0);
      us1 = unpack_u8_sse<1>(s1);
      us0 = _mm_sub_epi16(us0, us1);
      us0 = _mm_madd_epi16(us0, us0);
      acc32 = _mm_add_epi32(acc32, us0);
    }
    for (; x < w; ++x)
    {
      int d = (int)o1[y * O1 + x] - (int)o2[y * O2 + x];
      ssd += (d * d);
    }
  }

  acc32 = _mm_hadd_epi32(acc32, acc32);
  acc32 = _mm_hadd_epi32(acc32, acc32);

  ssd += (uint64_t)_mm_extract_epi32(acc32, 0);

  return ssd;
}

uint64_t ssd_10u_sse(
    const uint8_t *ref_ptr,
    const int ref_stride,
    const uint8_t *tar_ptr,
    const int tar_stride,
    const int w,
    const int h)
{
  constexpr int A = sizeof(__m128i) / 2;
  const int wA = w & ~(A - 1);
  const uint16_t *o1 = (const uint16_t *)ref_ptr;
  const uint16_t *o2 = (const uint16_t *)tar_ptr;
  const int O1 = ref_stride / 2;
  const int O2 = tar_stride / 2;

  __m128i acc64 = _mm_setzero_si128();
  uint64_t ssd = 0;

  for (int y = 0; y < h; ++y)
  {
    int x;
    for (x = 0; x < wA; x += A)
    {
      __m128i s0 = _mm_loadu_si128((__m128i *)&o1[y * O1 + x]);
      __m128i s1 = _mm_loadu_si128((__m128i *)&o2[y * O2 + x]);

      s0 = _mm_sub_epi16(s0, s1);
      s0 = _mm_madd_epi16(s0, s0);
      s0 = _mm_hadd_epi32(s0, s0);
      acc64 = _mm_add_epi64(acc64, unpack_u32_sse<0>(s0));
    }
    for (; x < w; ++x)
    {
      int d = (int)o1[y * O1 + x] - (int)o2[y * O2 + x];
      ssd += (d * d);
    }
  }

  ssd += (uint64_t)_mm_extract_epi64(acc64, 0);
  ssd += (uint64_t)_mm_extract_epi64(acc64, 1);

  return ssd;
}

uint64_t ssd_8u_avx2(
    const uint8_t *o1,
    const int O1,
    const uint8_t *o2,
    const int O2,
    const int w,
    const int h)
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
      __m256i s0 = _mm256_loadu_si256((__m256i *)&o1[y * O1 + x]);
      __m256i s1 = _mm256_loadu_si256((__m256i *)&o2[y * O2 + x]);

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
      int d = (int)o1[y * O1 + x] - (int)o2[y * O2 + x];
      ssd += (d * d);
    }
  }

  acc32 = _mm256_hadd_epi32(acc32, acc32);
  acc32 = _mm256_hadd_epi32(acc32, acc32);

  ssd += (uint64_t)_mm256_extract_epi32(acc32, 0);
  ssd += (uint64_t)_mm256_extract_epi32(acc32, 4);

  return ssd;
}

uint64_t ssd_10u_avx2(
    const uint8_t *ref_ptr,
    const int ref_stride,
    const uint8_t *tar_ptr,
    const int tar_stride,
    const int w,
    const int h)
{
  constexpr int A = sizeof(__m256i) / 2;
  const int wA = w & ~(A - 1);
  const uint16_t *o1 = (const uint16_t *)ref_ptr;
  const uint16_t *o2 = (const uint16_t *)tar_ptr;
  const int O1 = ref_stride / 2;
  const int O2 = tar_stride / 2;

  __m256i acc64 = _mm256_setzero_si256();
  uint64_t ssd = 0;

  for (int y = 0; y < h; ++y)
  {
    int x;
    for (x = 0; x < wA; x += A)
    {
      __m256i s0 = _mm256_loadu_si256((__m256i *)&o1[y * O1 + x]);
      __m256i s1 = _mm256_loadu_si256((__m256i *)&o2[y * O2 + x]);

      s0 = _mm256_sub_epi16(s0, s1);
      s0 = _mm256_madd_epi16(s0, s0);
      s0 = _mm256_hadd_epi32(s0, s0);
      acc64 = _mm256_add_epi64(acc64, unpack_u32_avx2<0>(s0));
    }
    for (; x < w; ++x)
    {
      int d = (int)o1[y * O1 + x] - (int)o2[y * O2 + x];
      ssd += (d * d);
    }
  }

  ssd += (uint64_t)_mm256_extract_epi64(acc64, 0);
  ssd += (uint64_t)_mm256_extract_epi64(acc64, 1);
  ssd += (uint64_t)_mm256_extract_epi64(acc64, 2);
  ssd += (uint64_t)_mm256_extract_epi64(acc64, 3);

  return ssd;
}

ssd_func_t get_ssd_func(
    const xpsnr_cpu_t cpu,
    int bit_depth)
{
  constexpr ssd_func_t TABLE[] = {
      &ssd_c<uint8_t>, &ssd_c<uint16_t>, &ssd_8u_sse, &ssd_10u_sse, &ssd_8u_avx2, &ssd_10u_avx2};

  assert(cpu >= xpsnr_cpu_c);

  int idx = ((int)cpu - (int)xpsnr_cpu_c) * 2 + (bit_depth > 8 ? 1 : 0);
  return TABLE[idx];
}