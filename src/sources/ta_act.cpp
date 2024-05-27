#include "ta_act.h"
#include "x86_common.h"
#include <algorithm>
#include <cmath>
#include <cassert>

template <typename T>
inline T clip3(const T val, const T min_val, const T max_val)
{
  return std::max(std::min(val, max_val), min_val);
}

template <typename T, int order, bool ds>
uint64_t diff_impl_c(
    const uint8_t *src,
    const uint8_t *src_m1,
    const uint8_t *src_m2,
    int src_stride,
    int wAct,
    int hAct)
{
  uint64_t taAct = 0;

  const T* o = (const T*)src;
  const T* oM1 = (const T*)src_m1;
  const T* oM2 = (const T*)src_m2;
  int O = src_stride / sizeof(T);

  if (!ds)
  {
    for (int y = 0; y < hAct; ++y)
    {
      for (int x = 0; x < wAct; ++x)
      {
        int t = 0;

        if (order == 1)
        {
          t = (int)o[y * O + x] - (int)oM1[y * O + x];
        }
        else
        {
          t = (int)o[y * O + x] - 2 * (int)oM1[y * O + x] + (int)oM2[y * O + x];
        }

        taAct += (uint64_t)std::abs(t);
      }
    }
  }
  else
  {
    for (int y = 0; y < hAct; y += 2)
    {
      for (int x = 0; x < wAct; x += 2)
      {
        int t = 0;

        if (order == 1)
        {
          t = (int)o[y * O + x] + (int)o[y * O + x + 1] + (int)o[(y + 1) * O + x] + (int)o[(y + 1) * O + x + 1] 
            - (int)oM1[y * O + x] - (int)oM1[y * O + x + 1] - (int)oM1[(y + 1) * O + x] - (int)oM1[(y + 1) * O + x + 1];
        }
        else
        {
          t = ((int)o[y * O + x] + (int)o[y * O + x + 1] + (int)o[(y + 1) * O + x] + (int)o[(y + 1) * O + x + 1])
              - 2 * ((int)oM1[y * O + x] + (int)oM1[y * O + x + 1] + (int)oM1[(y + 1) * O + x] + (int)oM1[(y + 1) * O + x + 1]) 
              + ((int)oM2[y * O + x] + (int)oM2[y * O + x + 1] + (int)oM2[(y + 1) * O + x] + (int)oM2[(y + 1) * O + x + 1]);
        }

        taAct += (uint64_t)std::abs(t);
      }
    }
  }

  return XPSNR_GAMMA * taAct;
}

namespace sse
{
  static uint64_t diff1st_8u_sse(
      const uint8_t *o,
      const uint8_t *oM1,
      const uint8_t *src_m2,
      int O,
      int wAct,
      int hAct)
  {
    (void)src_m2;

    const int A = sizeof(__m128i);
    const int wActA = (wAct & ~(A - 1));

    __m128i actSum32u = _mm_setzero_si128();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; ++y)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        __m128i lineM0 = _mm_loadu_si128((__m128i *)&o[y * O + x]); /* load 16 8-bit values */
        __m128i lineM1 = _mm_loadu_si128((__m128i *)&oM1[y * O + x]);

        __m128i sad16u = _mm_sad_epu8(lineM0, lineM1);
        actSum32u = _mm_add_epi32(actSum32u, sad16u);
      }
      for (; x < wAct; ++x)
      {
        const int t = (int)o[y * O + x] - (int)oM1[y * O + x];
        taAct += std::abs(t);
      }
    }
    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm_extract_epi32(actSum32u, 0);
    return taAct * XPSNR_GAMMA;
  }

  static uint64_t diff1st_10u_sse(
      const uint8_t *src,
      const uint8_t *src_m1,
      const uint8_t *src_m2,
      int src_stride,
      int wAct,
      int hAct)
  {
    (void)src_m2;
    constexpr int A = sizeof(__m128i) / 2;
    const int wActA = (wAct & ~(A - 1));
    const uint16_t *o = (const uint16_t *)src;
    const uint16_t *oM1 = (const uint16_t *)src_m1;
    const int O = src_stride / 2;

    __m128i actSum32u = _mm_setzero_si128();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; ++y)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        __m128i lineM0 = _mm_loadu_si128((__m128i *)&o[y * O + x]); /* load 8 16-bit values */
        __m128i lineM1 = _mm_loadu_si128((__m128i *)&oM1[y * O + x]);

        __m128i r = _mm_abs_epi16(_mm_sub_epi16(lineM0, lineM1));
        r = _mm_hadd_epi16(r, r);
        actSum32u = _mm_add_epi32(actSum32u, unpack_u16_sse<0>(r));
      }
      for (; x < wAct; ++x)
      {
        const int t = (int)o[y * O + x] - (int)oM1[y * O + x];
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm_extract_epi32(actSum32u, 0);

    return taAct * XPSNR_GAMMA;
  }

  template <int part>
  SIMD_INLINE __m128i laplacian(const __m128i a[3])
  {
    __m128i s0 = unpack_u8_sse<part>(a[0]);
    __m128i s1 = unpack_u8_sse<part>(a[1]);
    __m128i s2 = unpack_u8_sse<part>(a[2]);

    s0 = _mm_add_epi16(s0, s2);
    s1 = _mm_add_epi16(s1, s1);
    s0 = _mm_sub_epi16(s0, s1);

    return _mm_abs_epi16(s0);
  }

  static uint64_t diff2nd_8u_sse(
      const uint8_t *o,
      const uint8_t *oM1,
      const uint8_t *oM2,
      int O,
      int wAct,
      int hAct)
  {
    constexpr int A = sizeof(__m128i);
    const int wActA = wAct & ~(A - 1);

    __m128i a[3];
    __m128i actSum32u = _mm_setzero_si128();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; ++y)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        a[0] = _mm_loadu_si128((__m128i *)&o[y * O + x]); /* load 16 8-bit values */
        a[1] = _mm_loadu_si128((__m128i *)&oM1[y * O + x]);
        a[2] = _mm_loadu_si128((__m128i *)&oM2[y * O + x]);

        __m128i sum = _mm_add_epi16(laplacian<0>(a), laplacian<1>(a));
        sum = _mm_hadd_epi16(sum, sum);
        actSum32u = _mm_add_epi32(actSum32u, unpack_u16_sse<0>(sum));
      }
      for (; x < wAct; ++x)
      {
        int t = (int)o[y * O + x] - 2 * (int)oM1[y * O + x] + (int)oM2[y * O + x];
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm_extract_epi32(actSum32u, 0);

    return taAct * XPSNR_GAMMA;
  }

  static uint64_t diff2nd_10u_sse(
      const uint8_t *src,
      const uint8_t *src_m1,
      const uint8_t *src_m2,
      int src_stride,
      int wAct,
      int hAct)
  {
    constexpr int A = sizeof(__m128i) / 2;
    const int wActA = wAct & ~(A - 1);

    const uint16_t *o = (const uint16_t *)src;
    const uint16_t *oM1 = (const uint16_t *)src_m1;
    const uint16_t *oM2 = (const uint16_t *)src_m2;
    const int O = src_stride / 2;

    __m128i actSum32u = _mm_setzero_si128();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; ++y)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        __m128i s0 = _mm_loadu_si128((__m128i *)&o[y * O + x]); /* load 8 16-bit values */
        __m128i s1 = _mm_loadu_si128((__m128i *)&oM1[y * O + x]);
        __m128i s2 = _mm_loadu_si128((__m128i *)&oM2[y * O + x]);
        s0 = _mm_add_epi16(s0, s2);
        s1 = _mm_add_epi16(s1, s1);
        s0 = _mm_sub_epi16(s0, s1);
        s0 = _mm_abs_epi16(s0);
        s0 = _mm_hadd_epi16(s0, s0);

        actSum32u = _mm_add_epi32(actSum32u, unpack_u16_sse<0>(s0));
      }
      for (; x < wAct; ++x)
      {
        int t = (int)o[y * O + x] - 2 * (int)oM1[y * O + x] + (int)oM2[y * O + x];
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm_extract_epi32(actSum32u, 0);

    return taAct * XPSNR_GAMMA;
  }

  template <int part>
  SIMD_INLINE __m128i ds_grad(const __m128i a[4])
  {
    __m128i m1 = _mm_add_epi16(unpack_u8_sse<part>(a[0]), unpack_u8_sse<part>(a[1]));
    __m128i m2 = _mm_add_epi16(unpack_u8_sse<part>(a[2]), unpack_u8_sse<part>(a[3]));

    m1 = _mm_sub_epi16(m1, m2);
    m1 = _mm_hadd_epi16(m1, m1);
    return _mm_abs_epi16(m1);
  }

  static uint64_t diff1st_ds_8u_sse(
      const uint8_t *o,
      const uint8_t *oM1,
      const uint8_t *src_m2,
      int O,
      int wAct,
      int hAct)
  {
    (void)src_m2;
    constexpr int A = sizeof(__m128i);
    const int wActA = wAct & ~(A - 1);

    __m128i actSum32u = _mm_setzero_si128();
    __m128i a[4];
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; y += 2)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        a[0] = _mm_loadu_si128((__m128i *)&o[y * O + x]); /* load 16 8-bit values */
        a[1] = _mm_loadu_si128((__m128i *)&o[(y + 1) * O + x]);
        a[2] = _mm_loadu_si128((__m128i *)&oM1[y * O + x]);
        a[3] = _mm_loadu_si128((__m128i *)&oM1[(y + 1) * O + x]);

        __m128i lo = ds_grad<0>(a);
        actSum32u = _mm_add_epi32(actSum32u, unpack_u16_sse<0>(lo));

        __m128i hi = ds_grad<1>(a);
        actSum32u = _mm_add_epi32(actSum32u, unpack_u16_sse<0>(hi));
      }
      for (; x < wAct; x += 2)
      {
        int t = (int)o[y * O + x] + (int)o[y * O + x + 1] + (int)o[(y + 1) * O + x] + (int)o[(y + 1) * O + x + 1] - (int)oM1[y * O + x] - (int)oM1[y * O + x + 1] - (int)oM1[(y + 1) * O + x] - (int)oM1[(y + 1) * O + x + 1];
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm_extract_epi32(actSum32u, 0);

    return taAct * XPSNR_GAMMA;
  }

  static uint64_t diff1st_ds_10u_sse(
      const uint8_t *src,
      const uint8_t *src_m1,
      const uint8_t *src_m2,
      int src_stride,
      int wAct,
      int hAct)
  {
    (void)src_m2;

    constexpr int A = sizeof(__m128i) / 2;
    const int wActA = wAct & ~(A - 1);

    const uint16_t *o = (const uint16_t *)src;
    const uint16_t *oM1 = (const uint16_t *)src_m1;
    const int O = src_stride / 2;

    __m128i a[4];
    __m128i actSum32u = _mm_setzero_si128();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; y += 2)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        a[0] = _mm_loadu_si128((__m128i *)&o[y * O + x]); /* load 8 16-bit values */
        a[1] = _mm_loadu_si128((__m128i *)&o[(y + 1) * O + x]);
        a[2] = _mm_loadu_si128((__m128i *)&oM1[y * O + x]);
        a[3] = _mm_loadu_si128((__m128i *)&oM1[(y + 1) * O + x]);

        __m128i m1 = _mm_add_epi16(a[0], a[1]);
        __m128i m2 = _mm_add_epi16(a[2], a[3]);
        m1 = _mm_sub_epi16(m1, m2);
        m1 = _mm_hadd_epi16(m1, m1);
        m1 = _mm_abs_epi16(m1);
        actSum32u = _mm_add_epi32(actSum32u, unpack_u16_sse<0>(m1));
      }
      for (; x < wAct; x += 2)
      {
        int t = (int)o[y * O + x] + (int)o[y * O + x + 1] + (int)o[(y + 1) * O + x] + (int)o[(y + 1) * O + x + 1] - (int)oM1[y * O + x] - (int)oM1[y * O + x + 1] - (int)oM1[(y + 1) * O + x] - (int)oM1[(y + 1) * O + x + 1];
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm_extract_epi32(actSum32u, 0);

    return taAct * XPSNR_GAMMA;
  }

  template <int part>
  SIMD_INLINE __m128i ds_laplacian(const __m128i a[6])
  {
    __m128i m1 = _mm_add_epi16(unpack_u8_sse<part>(a[0]), unpack_u8_sse<part>(a[1]));
    __m128i m2 = _mm_add_epi16(unpack_u8_sse<part>(a[2]), unpack_u8_sse<part>(a[3]));
    __m128i m3 = _mm_add_epi16(unpack_u8_sse<part>(a[4]), unpack_u8_sse<part>(a[5]));

    m1 = _mm_add_epi16(m1, m3);
    m2 = _mm_add_epi16(m2, m2);
    m1 = _mm_sub_epi16(m1, m2);
    m1 = _mm_hadd_epi16(m1, m1);
    return _mm_abs_epi16(m1);
  }

  static uint64_t diff2nd_ds_8u_sse(
      const uint8_t *o,
      const uint8_t *oM1,
      const uint8_t *oM2,
      int O,
      int wAct,
      int hAct)
  {
    constexpr int A = sizeof(__m128i);
    const int wActA = wAct & ~(A - 1);

    __m128i a[6];
    __m128i actSum32u = _mm_setzero_si128();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; y += 2)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        a[0] = _mm_loadu_si128((__m128i *)&o[y * O + x]); /* load 16 8-bit values */
        a[1] = _mm_loadu_si128((__m128i *)&o[(y + 1) * O + x]);
        a[2] = _mm_loadu_si128((__m128i *)&oM1[y * O + x]);
        a[3] = _mm_loadu_si128((__m128i *)&oM1[(y + 1) * O + x]);
        a[4] = _mm_loadu_si128((__m128i *)&oM2[y * O + x]);
        a[5] = _mm_loadu_si128((__m128i *)&oM2[(y + 1) * O + x]);

        __m128i lo = ds_laplacian<0>(a);
        actSum32u = _mm_add_epi32(actSum32u, unpack_u16_sse<0>(lo));
        __m128i hi = ds_laplacian<1>(a);
        actSum32u = _mm_add_epi32(actSum32u, unpack_u16_sse<0>(hi));
      }
      for (; x < wAct; x += 2)
      {
        int t = ((int)o[y * O + x] + (int)o[y * O + x + 1] + (int)o[(y + 1) * O + x] + (int)o[(y + 1) * O + x + 1]) - 2 * ((int)oM1[y * O + x] + (int)oM1[y * O + x + 1] + (int)oM1[(y + 1) * O + x] + (int)oM1[(y + 1) * O + x + 1]) + ((int)oM2[y * O + x] + (int)oM2[y * O + x + 1] + (int)oM2[(y + 1) * O + x] + (int)oM2[(y + 1) * O + x + 1]);
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm_extract_epi32(actSum32u, 0);

    return taAct * XPSNR_GAMMA;
  }

  static uint64_t diff2nd_ds_10u_sse(
      const uint8_t *src,
      const uint8_t *src_m1,
      const uint8_t *src_m2,
      int src_stride,
      int wAct,
      int hAct)
  {
    constexpr int A = sizeof(__m128i) / 2;
    const int wActA = wAct & ~(A - 1);

    const uint16_t *o = (const uint16_t *)src;
    const uint16_t *oM1 = (const uint16_t *)src_m1;
    const uint16_t *oM2 = (const uint16_t *)src_m2;
    const int O = src_stride / 2;

    __m128i actSum32u = _mm_setzero_si128();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; y += 2)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        __m128i lineM0u = _mm_loadu_si128((__m128i *)&o[y * O + x]); /* load 8 16-bit values */
        __m128i lineM0d = _mm_loadu_si128((__m128i *)&o[(y + 1) * O + x]);
        __m128i lineM1u = _mm_loadu_si128((__m128i *)&oM1[y * O + x]);
        __m128i lineM1d = _mm_loadu_si128((__m128i *)&oM1[(y + 1) * O + x]);
        __m128i lineM2u = _mm_loadu_si128((__m128i *)&oM2[y * O + x]);
        __m128i lineM2d = _mm_loadu_si128((__m128i *)&oM2[(y + 1) * O + x]);

        __m128i M0 = _mm_add_epi16(lineM0u, lineM0d);
        __m128i M1 = _mm_add_epi16(lineM1u, lineM1d);
        __m128i M2 = _mm_add_epi16(lineM2u, lineM2d);

        M0 = _mm_add_epi16(M0, M2);
        M1 = _mm_add_epi16(M1, M1);
        M0 = _mm_sub_epi16(M0, M1);
        M0 = _mm_hadd_epi16(M0, M0);
        M0 = _mm_abs_epi16(M0);

        actSum32u = _mm_add_epi32(actSum32u, unpack_u16_sse<0>(M0));
      }
      for (; x < wAct; x += 2)
      {
        int t = ((int)o[y * O + x] + (int)o[y * O + x + 1] + (int)o[(y + 1) * O + x] + (int)o[(y + 1) * O + x + 1]) - 2 * ((int)oM1[y * O + x] + (int)oM1[y * O + x + 1] + (int)oM1[(y + 1) * O + x] + (int)oM1[(y + 1) * O + x + 1]) + ((int)oM2[y * O + x] + (int)oM2[y * O + x + 1] + (int)oM2[(y + 1) * O + x] + (int)oM2[(y + 1) * O + x + 1]);
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm_extract_epi32(actSum32u, 0);

    return taAct * XPSNR_GAMMA;
  }
}

namespace avx2
{
  static uint64_t diff1st_8u_avx2(
      const uint8_t *o,
      const uint8_t *oM1,
      const uint8_t *src_m2,
      int O,
      int wAct,
      int hAct)
  {
    (void)src_m2;

    const int A = sizeof(__m256i);
    const int wActA = (wAct & ~(A - 1));

    __m256i actSum32u = _mm256_setzero_si256();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; ++y)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        __m256i lineM0 = _mm256_loadu_si256((__m256i *)&o[y * O + x]); /* load 16 8-bit values */
        __m256i lineM1 = _mm256_loadu_si256((__m256i *)&oM1[y * O + x]);

        __m256i sad16u = _mm256_sad_epu8(lineM0, lineM1);
        actSum32u = _mm256_add_epi32(actSum32u, sad16u);
      }
      for (; x < wAct; ++x)
      {
        const int t = (int)o[y * O + x] - (int)oM1[y * O + x];
        taAct += std::abs(t);
      }
    }
    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 0);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 4);
    return taAct * XPSNR_GAMMA;
  }

  static uint64_t diff1st_10u_avx2(
      const uint8_t *src,
      const uint8_t *src_m1,
      const uint8_t *src_m2,
      int src_stride,
      int wAct,
      int hAct)
  {
    (void)src_m2;
    constexpr int A = sizeof(__m256i) / 2;
    const int wActA = (wAct & ~(A - 1));
    const uint16_t *o = (const uint16_t *)src;
    const uint16_t *oM1 = (const uint16_t *)src_m1;
    const int O = src_stride / 2;

    __m256i actSum32u = _mm256_setzero_si256();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; ++y)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        __m256i lineM0 = _mm256_loadu_si256((__m256i *)&o[y * O + x]); /* load 8 16-bit values */
        __m256i lineM1 = _mm256_loadu_si256((__m256i *)&oM1[y * O + x]);

        __m256i r = _mm256_abs_epi16(_mm256_sub_epi16(lineM0, lineM1));
        r = _mm256_hadd_epi16(r, r);
        actSum32u = _mm256_add_epi32(actSum32u, unpack_u16_avx2<0>(r));
      }
      for (; x < wAct; ++x)
      {
        const int t = (int)o[y * O + x] - (int)oM1[y * O + x];
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 0);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 4);

    return taAct * XPSNR_GAMMA;
  }

  template <int part>
  SIMD_INLINE __m256i laplacian(const __m256i a[3])
  {
    __m256i s0 = unpack_u8_avx2<part>(a[0]);
    __m256i s1 = unpack_u8_avx2<part>(a[1]);
    __m256i s2 = unpack_u8_avx2<part>(a[2]);

    s0 = _mm256_add_epi16(s0, s2);
    s1 = _mm256_add_epi16(s1, s1);
    s0 = _mm256_sub_epi16(s0, s1);

    return _mm256_abs_epi16(s0);
  }

  static uint64_t diff2nd_8u_avx2(
      const uint8_t *o,
      const uint8_t *oM1,
      const uint8_t *oM2,
      int O,
      int wAct,
      int hAct)
  {
    constexpr int A = sizeof(__m256i);
    const int wActA = wAct & ~(A - 1);

    __m256i a[3];
    __m256i actSum32u = _mm256_setzero_si256();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; ++y)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        a[0] = _mm256_loadu_si256((__m256i *)&o[y * O + x]); /* load 32 8-bit values */
        a[1] = _mm256_loadu_si256((__m256i *)&oM1[y * O + x]);
        a[2] = _mm256_loadu_si256((__m256i *)&oM2[y * O + x]);

        __m256i sum = _mm256_add_epi16(laplacian<0>(a), laplacian<1>(a));
        sum = _mm256_hadd_epi16(sum, sum);
        actSum32u = _mm256_add_epi32(actSum32u, unpack_u16_avx2<0>(sum));
      }
      for (; x < wAct; ++x)
      {
        int t = (int)o[y * O + x] - 2 * (int)oM1[y * O + x] + (int)oM2[y * O + x];
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 0);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 4);

    return taAct * XPSNR_GAMMA;
  }

  static uint64_t diff2nd_10u_avx2(
      const uint8_t *src,
      const uint8_t *src_m1,
      const uint8_t *src_m2,
      int src_stride,
      int wAct,
      int hAct)
  {
    constexpr int A = sizeof(__m256i) / 2;
    const int wActA = wAct & ~(A - 1);

    const uint16_t *o = (const uint16_t *)src;
    const uint16_t *oM1 = (const uint16_t *)src_m1;
    const uint16_t *oM2 = (const uint16_t *)src_m2;
    const int O = src_stride / 2;

    __m256i actSum32u = _mm256_setzero_si256();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; ++y)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        __m256i s0 = _mm256_loadu_si256((__m256i *)&o[y * O + x]); /* load 8 16-bit values */
        __m256i s1 = _mm256_loadu_si256((__m256i *)&oM1[y * O + x]);
        __m256i s2 = _mm256_loadu_si256((__m256i *)&oM2[y * O + x]);
        s0 = _mm256_add_epi16(s0, s2);
        s1 = _mm256_add_epi16(s1, s1);
        s0 = _mm256_sub_epi16(s0, s1);
        s0 = _mm256_abs_epi16(s0);
        s0 = _mm256_hadd_epi16(s0, s0);

        actSum32u = _mm256_add_epi32(actSum32u, unpack_u16_avx2<0>(s0));
      }
      for (; x < wAct; ++x)
      {
        int t = (int)o[y * O + x] - 2 * (int)oM1[y * O + x] + (int)oM2[y * O + x];
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 0);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 4);

    return taAct * XPSNR_GAMMA;
  }

  template <int part>
  SIMD_INLINE __m256i ds_grad(const __m256i a[4])
  {
    __m256i m1 = _mm256_add_epi16(unpack_u8_avx2<part>(a[0]), unpack_u8_avx2<part>(a[1]));
    __m256i m2 = _mm256_add_epi16(unpack_u8_avx2<part>(a[2]), unpack_u8_avx2<part>(a[3]));

    m1 = _mm256_sub_epi16(m1, m2);
    m1 = _mm256_hadd_epi16(m1, m1);
    return _mm256_abs_epi16(m1);
  }

  static uint64_t diff1st_ds_8u_avx2(
      const uint8_t *o,
      const uint8_t *oM1,
      const uint8_t *src_m2,
      int O,
      int wAct,
      int hAct)
  {
    (void)src_m2;
    constexpr int A = sizeof(__m256i);
    const int wActA = wAct & ~(A - 1);

    __m256i actSum32u = _mm256_setzero_si256();
    __m256i a[4];
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; y += 2)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        a[0] = _mm256_loadu_si256((__m256i *)&o[y * O + x]); /* load 32 8-bit values */
        a[1] = _mm256_loadu_si256((__m256i *)&o[(y + 1) * O + x]);
        a[2] = _mm256_loadu_si256((__m256i *)&oM1[y * O + x]);
        a[3] = _mm256_loadu_si256((__m256i *)&oM1[(y + 1) * O + x]);

        __m256i lo = ds_grad<0>(a);
        actSum32u = _mm256_add_epi32(actSum32u, unpack_u16_avx2<0>(lo));

        __m256i hi = ds_grad<1>(a);
        actSum32u = _mm256_add_epi32(actSum32u, unpack_u16_avx2<0>(hi));
      }
      for (; x < wAct; x += 2)
      {
        int t = (int)o[y * O + x] + (int)o[y * O + x + 1] + (int)o[(y + 1) * O + x] + (int)o[(y + 1) * O + x + 1] - (int)oM1[y * O + x] - (int)oM1[y * O + x + 1] - (int)oM1[(y + 1) * O + x] - (int)oM1[(y + 1) * O + x + 1];
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 0);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 4);

    return taAct * XPSNR_GAMMA;
  }

  static uint64_t diff1st_ds_10u_avx2(
      const uint8_t *src,
      const uint8_t *src_m1,
      const uint8_t *src_m2,
      int src_stride,
      int wAct,
      int hAct)
  {
    (void)src_m2;

    constexpr int A = sizeof(__m256i) / 2;
    const int wActA = wAct & ~(A - 1);

    const uint16_t *o = (const uint16_t *)src;
    const uint16_t *oM1 = (const uint16_t *)src_m1;
    const int O = src_stride / 2;

    __m256i a[4];
    __m256i actSum32u = _mm256_setzero_si256();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; y += 2)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        a[0] = _mm256_loadu_si256((__m256i *)&o[y * O + x]); /* load 16 16-bit values */
        a[1] = _mm256_loadu_si256((__m256i *)&o[(y + 1) * O + x]);
        a[2] = _mm256_loadu_si256((__m256i *)&oM1[y * O + x]);
        a[3] = _mm256_loadu_si256((__m256i *)&oM1[(y + 1) * O + x]);

        __m256i m1 = _mm256_add_epi16(a[0], a[1]);
        __m256i m2 = _mm256_add_epi16(a[2], a[3]);
        m1 = _mm256_sub_epi16(m1, m2);
        m1 = _mm256_hadd_epi16(m1, m1);
        m1 = _mm256_abs_epi16(m1);
        actSum32u = _mm256_add_epi32(actSum32u, unpack_u16_avx2<0>(m1));
      }
      for (; x < wAct; x += 2)
      {
        int t = (int)o[y * O + x] + (int)o[y * O + x + 1] + (int)o[(y + 1) * O + x] + (int)o[(y + 1) * O + x + 1] - (int)oM1[y * O + x] - (int)oM1[y * O + x + 1] - (int)oM1[(y + 1) * O + x] - (int)oM1[(y + 1) * O + x + 1];
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 0);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 4);

    return taAct * XPSNR_GAMMA;
  }

  template <int part>
  SIMD_INLINE __m256i ds_laplacian(const __m256i a[6])
  {
    __m256i m1 = _mm256_add_epi16(unpack_u8_avx2<part>(a[0]), unpack_u8_avx2<part>(a[1]));
    __m256i m2 = _mm256_add_epi16(unpack_u8_avx2<part>(a[2]), unpack_u8_avx2<part>(a[3]));
    __m256i m3 = _mm256_add_epi16(unpack_u8_avx2<part>(a[4]), unpack_u8_avx2<part>(a[5]));

    m1 = _mm256_add_epi16(m1, m3);
    m2 = _mm256_add_epi16(m2, m2);
    m1 = _mm256_sub_epi16(m1, m2);
    m1 = _mm256_hadd_epi16(m1, m1);
    return _mm256_abs_epi16(m1);
  }

  static uint64_t diff2nd_ds_8u_avx2(
      const uint8_t *o,
      const uint8_t *oM1,
      const uint8_t *oM2,
      int O,
      int wAct,
      int hAct)
  {
    constexpr int A = sizeof(__m256i);
    const int wActA = wAct & ~(A - 1);

    __m256i a[6];
    __m256i actSum32u = _mm256_setzero_si256();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; y += 2)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        a[0] = _mm256_loadu_si256((__m256i *)&o[y * O + x]); /* load 32 8-bit values */
        a[1] = _mm256_loadu_si256((__m256i *)&o[(y + 1) * O + x]);
        a[2] = _mm256_loadu_si256((__m256i *)&oM1[y * O + x]);
        a[3] = _mm256_loadu_si256((__m256i *)&oM1[(y + 1) * O + x]);
        a[4] = _mm256_loadu_si256((__m256i *)&oM2[y * O + x]);
        a[5] = _mm256_loadu_si256((__m256i *)&oM2[(y + 1) * O + x]);

        __m256i lo = ds_laplacian<0>(a);
        actSum32u = _mm256_add_epi32(actSum32u, unpack_u16_avx2<0>(lo));
        __m256i hi = ds_laplacian<1>(a);
        actSum32u = _mm256_add_epi32(actSum32u, unpack_u16_avx2<0>(hi));
      }
      for (; x < wAct; x += 2)
      {
        int t = ((int)o[y * O + x] + (int)o[y * O + x + 1] + (int)o[(y + 1) * O + x] + (int)o[(y + 1) * O + x + 1]) - 2 * ((int)oM1[y * O + x] + (int)oM1[y * O + x + 1] + (int)oM1[(y + 1) * O + x] + (int)oM1[(y + 1) * O + x + 1]) + ((int)oM2[y * O + x] + (int)oM2[y * O + x + 1] + (int)oM2[(y + 1) * O + x] + (int)oM2[(y + 1) * O + x + 1]);
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 0);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 4);

    return taAct * XPSNR_GAMMA;
  }

  static uint64_t diff2nd_ds_10u_avx2(
      const uint8_t *src,
      const uint8_t *src_m1,
      const uint8_t *src_m2,
      int src_stride,
      int wAct,
      int hAct)
  {
    constexpr int A = sizeof(__m256i) / 2;
    const int wActA = wAct & ~(A - 1);

    const uint16_t *o = (const uint16_t *)src;
    const uint16_t *oM1 = (const uint16_t *)src_m1;
    const uint16_t *oM2 = (const uint16_t *)src_m2;
    const int O = src_stride / 2;

    __m256i actSum32u = _mm256_setzero_si256();
    uint64_t taAct = 0;

    for (int y = 0; y < hAct; y += 2)
    {
      int x;
      for (x = 0; x < wActA; x += A)
      {
        __m256i lineM0u = _mm256_loadu_si256((__m256i *)&o[y * O + x]); /* load 8 16-bit values */
        __m256i lineM0d = _mm256_loadu_si256((__m256i *)&o[(y + 1) * O + x]);
        __m256i lineM1u = _mm256_loadu_si256((__m256i *)&oM1[y * O + x]);
        __m256i lineM1d = _mm256_loadu_si256((__m256i *)&oM1[(y + 1) * O + x]);
        __m256i lineM2u = _mm256_loadu_si256((__m256i *)&oM2[y * O + x]);
        __m256i lineM2d = _mm256_loadu_si256((__m256i *)&oM2[(y + 1) * O + x]);

        __m256i M0 = _mm256_add_epi16(lineM0u, lineM0d);
        __m256i M1 = _mm256_add_epi16(lineM1u, lineM1d);
        __m256i M2 = _mm256_add_epi16(lineM2u, lineM2d);

        M0 = _mm256_add_epi16(M0, M2);
        M1 = _mm256_add_epi16(M1, M1);
        M0 = _mm256_sub_epi16(M0, M1);
        M0 = _mm256_hadd_epi16(M0, M0);
        M0 = _mm256_abs_epi16(M0);

        actSum32u = _mm256_add_epi32(actSum32u, unpack_u16_avx2<0>(M0));
      }
      for (; x < wAct; x += 2)
      {
        int t = ((int)o[y * O + x] + (int)o[y * O + x + 1] + (int)o[(y + 1) * O + x] + (int)o[(y + 1) * O + x + 1]) - 2 * ((int)oM1[y * O + x] + (int)oM1[y * O + x + 1] + (int)oM1[(y + 1) * O + x] + (int)oM1[(y + 1) * O + x + 1]) + ((int)oM2[y * O + x] + (int)oM2[y * O + x + 1] + (int)oM2[(y + 1) * O + x] + (int)oM2[(y + 1) * O + x + 1]);
        taAct += std::abs(t);
      }
    }

    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    actSum32u = _mm256_hadd_epi32(actSum32u, actSum32u);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 0);
    taAct += (uint64_t)_mm256_extract_epi32(actSum32u, 4);

    return taAct * XPSNR_GAMMA;
  }
}

temp_act_func_t get_temp_act_func(
    xpsnr_cpu_t cpu,
    const int bit_depth,
    const int diff_order, // 1 or 2
    bool down_sampling)
{
  // [ds][order][bd]
  constexpr temp_act_func_t TABLE[] = {
      diff_impl_c<uint8_t, 1, false>,
      diff_impl_c<uint16_t, 1, false>,
      diff_impl_c<uint8_t, 2, false>,
      diff_impl_c<uint16_t, 2, false>,
      diff_impl_c<uint8_t, 1, true>,
      diff_impl_c<uint16_t, 1, true>,
      diff_impl_c<uint8_t, 2, true>,
      diff_impl_c<uint16_t, 2, true>,
      sse::diff1st_8u_sse,
      sse::diff1st_10u_sse,
      sse::diff2nd_8u_sse,
      sse::diff2nd_10u_sse,
      sse::diff1st_ds_8u_sse,
      sse::diff1st_ds_10u_sse,
      sse::diff2nd_ds_8u_sse,
      sse::diff2nd_ds_10u_sse,
      avx2::diff1st_8u_avx2,
      avx2::diff1st_10u_avx2,
      avx2::diff2nd_8u_avx2,
      avx2::diff2nd_10u_avx2,
      avx2::diff1st_ds_8u_avx2,
      avx2::diff1st_ds_10u_avx2,
      avx2::diff2nd_ds_8u_avx2,
      avx2::diff2nd_ds_10u_avx2,
  };

  assert(cpu >= xpsnr_cpu_c);
  assert(diff_order == 1 || diff_order == 2);

  int idx = ((int)cpu - (int)xpsnr_cpu_c) * 8 +
            (down_sampling ? 4 : 0) +
            (diff_order == 2 ? 2 : 0) +
            (bit_depth > 8 ? 1 : 0);

  return TABLE[idx];
}