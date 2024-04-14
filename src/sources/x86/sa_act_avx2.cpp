#include "sa_act.h"
#include "x86_common.h"
#include <x86intrin.h>
#include <cmath>

template <typename T>
static SIMD_INLINE void load_body9(const T *o, int O, __m256i a[9])
{
  a[0] = _mm256_loadu_si256((__m256i *)&o[-O - 1]);
  a[1] = _mm256_loadu_si256((__m256i *)&o[-O]);
  a[2] = _mm256_loadu_si256((__m256i *)&o[-O + 1]);
  a[3] = _mm256_loadu_si256((__m256i *)&o[-1]);
  a[4] = _mm256_loadu_si256((__m256i *)&o[0]);
  a[5] = _mm256_loadu_si256((__m256i *)&o[1]);
  a[6] = _mm256_loadu_si256((__m256i *)&o[O - 1]);
  a[7] = _mm256_loadu_si256((__m256i *)&o[O]);
  a[8] = _mm256_loadu_si256((__m256i *)&o[O + 1]);
}

template <int part>
SIMD_INLINE __m256i highpass_op8u(__m256i a[9])
{
  // Kerenel shape:
  // −1 −2 −1
  // −2 12 −2
  // −1 −2 −1

  __m256i s[9];

  for (int i = 0; i < 9; ++i)
  {
    s[i] = unpack_u8_avx2<part>(a[i]);
  }

  __m256i center = _mm256_add_epi16(_mm256_slli_epi16(s[4], 3), _mm256_slli_epi16(s[4], 2));
  __m256i side = _mm256_add_epi16(_mm256_add_epi16(s[1], s[3]), _mm256_add_epi16(s[5], s[7]));
  side = _mm256_slli_epi16(side, 1);
  __m256i diag = _mm256_add_epi16(_mm256_add_epi16(s[0], s[2]), _mm256_add_epi16(s[6], s[8]));
  side = _mm256_add_epi16(side, diag);

  center = _mm256_sub_epi16(center, side);
  return _mm256_abs_epi16(center);
}

SIMD_INLINE __m256i highpass_op10u(const __m256i a[9])
{
  // Kerenel shape:
  // −1 −2 −1
  // −2 12 −2
  // −1 −2 −1

  __m256i center = _mm256_add_epi16(_mm256_slli_epi16(a[4], 3), _mm256_slli_epi16(a[4], 2));
  
  __m256i side = _mm256_add_epi16(_mm256_add_epi16(a[1], a[3]), _mm256_add_epi16(a[5], a[7]));
  side = _mm256_slli_epi16(side, 1);
  center = _mm256_sub_epi16(center, side);

  side = _mm256_add_epi16(_mm256_add_epi16(a[0], a[2]), _mm256_add_epi16(a[6], a[8]));
  center = _mm256_sub_epi16(center, side);

  return _mm256_abs_epi16(center);
}

static uint64_t high8u_avx2(
    const uint8_t *o,
    int O,
    int xAct,
    int yAct,
    int wAct,
    int hAct)
{
  constexpr int A = (int)sizeof(__m256i);
  const int wActA = ((wAct - xAct) & ~(A - 1)) + xAct;

  __m256i a[9];
  __m256i actSum = _mm256_setzero_si256();
  uint64_t saAct = 0u;
  
  for (int y = yAct; y < hAct; ++y)
  {
    int x;
    for (x = xAct; x < wActA; x += A)
    {
      load_body9(&o[y * O + x], O, a);

      __m256i slo = highpass_op8u<0>(a);
      __m256i shi = highpass_op8u<1>(a);

      slo = _mm256_add_epi16(slo, shi);
      slo = _mm256_hadd_epi16(slo, slo);
      actSum = _mm256_add_epi32(actSum, unpack_u16_avx2<0>(slo));
    }
    for(; x < wAct; ++x)
    {
      const int f = 12 * (int)o[y * O + x] 
        - 2 * ( (int)o[y * O + x - 1] + (int)o[y * O + x + 1] + (int)o[(y - 1) * O + x] + (int)o[(y + 1) * O + x] ) 
        - ( (int)o[(y - 1) * O + x - 1] + (int)o[(y - 1) * O + x + 1] + (int)o[(y + 1) * O + x - 1] + (int)o[(y + 1) * O + x + 1]);
      saAct += (uint64_t)std::abs(f);
    }
  }

  actSum = _mm256_hadd_epi32(actSum, actSum);
  actSum = _mm256_hadd_epi32(actSum, actSum);

  saAct += _mm256_extract_epi32(actSum, 0);
  saAct += _mm256_extract_epi32(actSum, 4);

  return saAct;
}

static uint64_t high10u_avx2(
    const uint8_t *src,
    int src_stride,
    int xAct,
    int yAct,
    int wAct,
    int hAct)
{
  constexpr int A = sizeof(__m256i) / 2;
  const int wActA = ((wAct - xAct) & ~(A - 1)) + xAct;
  const uint16_t *o = (const uint16_t *)src;
  const int O = src_stride / sizeof(uint16_t);

  __m256i a[9];
  __m256i actSum = _mm256_setzero_si256();
  uint64_t saAct = 0u;

  for (int y = yAct; y < hAct; ++y)
  {
    int x;
    for (x = xAct; x < wActA; x += A)
    {
      load_body9(&o[y * O + x], O, a);
      __m256i sum = highpass_op10u(a);
      sum = _mm256_hadd_epi16(sum, sum);
      actSum = _mm256_add_epi32(actSum, unpack_u16_avx2<0>(sum));
    }
    for(; x < wAct; ++x)
    {
      int f = 12 * (int)o[y * O + x] 
        - 2 * ( (int)o[y * O + x - 1] + (int)o[y * O + x + 1] + (int)o[(y - 1) * O + x] + (int)o[(y + 1) * O + x] ) 
        - ( (int)o[(y - 1) * O + x - 1] + (int)o[(y - 1) * O + x + 1] + (int)o[(y + 1) * O + x - 1] + (int)o[(y + 1) * O + x + 1]);

      saAct += std::abs(f);
    }
  }

  actSum = _mm256_hadd_epi32(actSum, actSum);
  actSum = _mm256_hadd_epi32(actSum, actSum);
  saAct += _mm256_extract_epi32(actSum, 0);
  saAct += _mm256_extract_epi32(actSum, 4);

  return saAct;
}

template <int part, typename T>
static SIMD_INLINE void load_line2(const T *o, int O, __m256i row1[6], __m256i row2[6])
{
  const __m256i bitmask = (sizeof(T) == 1) ? _mm256_set1_epi16(0x00FF) : _mm256_set1_epi32(0x0000FFFF);

  if (part == 0)
  {
    row1[0] = _mm256_loadu_si256((__m256i *)&o[-O * 2 - 1]);
    row1[1] = _mm256_loadu_si256((__m256i *)&o[-O * 2]);
    row1[2] = _mm256_loadu_si256((__m256i *)&o[-O * 2 + 1]);
    row1[3] = _mm256_loadu_si256((__m256i *)&o[-O * 2 + 2]);

    row2[0] = _mm256_loadu_si256((__m256i *)&o[O * 3 - 1]);
    row2[1] = _mm256_loadu_si256((__m256i *)&o[O * 3]);
    row2[2] = _mm256_loadu_si256((__m256i *)&o[O * 3 + 1]);
    row2[3] = _mm256_loadu_si256((__m256i *)&o[O * 3 + 2]);
  }
  else if (part == 1)
  {
    row1[0] = _mm256_loadu_si256((__m256i *)&o[-O - 2]);
    row1[1] = _mm256_loadu_si256((__m256i *)&o[-O - 1]);
    row1[2] = _mm256_loadu_si256((__m256i *)&o[-O]);
    row1[3] = _mm256_loadu_si256((__m256i *)&o[-O + 1]);
    row1[4] = _mm256_loadu_si256((__m256i *)&o[-O + 2]);
    row1[5] = _mm256_loadu_si256((__m256i *)&o[-O + 3]);

    row2[0] = _mm256_loadu_si256((__m256i *)&o[O * 2 - 2]);
    row2[1] = _mm256_loadu_si256((__m256i *)&o[O * 2 - 1]);
    row2[2] = _mm256_loadu_si256((__m256i *)&o[O * 2]);
    row2[3] = _mm256_loadu_si256((__m256i *)&o[O * 2 + 1]);
    row2[4] = _mm256_loadu_si256((__m256i *)&o[O * 2 + 2]);
    row2[5] = _mm256_loadu_si256((__m256i *)&o[O * 2 + 3]);
  }
  else
  {
    row1[0] = _mm256_loadu_si256((__m256i *)&o[-2]);
    row1[1] = _mm256_loadu_si256((__m256i *)&o[-1]);
    row1[2] = _mm256_loadu_si256((__m256i *)&o[0]);
    row1[3] = _mm256_loadu_si256((__m256i *)&o[1]);
    row1[4] = _mm256_loadu_si256((__m256i *)&o[2]);
    row1[5] = _mm256_loadu_si256((__m256i *)&o[3]);

    row2[0] = _mm256_loadu_si256((__m256i *)&o[O - 2]);
    row2[1] = _mm256_loadu_si256((__m256i *)&o[O - 1]);
    row2[2] = _mm256_loadu_si256((__m256i *)&o[O]);
    row2[3] = _mm256_loadu_si256((__m256i *)&o[O + 1]);
    row2[4] = _mm256_loadu_si256((__m256i *)&o[O + 2]);
    row2[5] = _mm256_loadu_si256((__m256i *)&o[O + 3]);
  }
  
  row1[0] = _mm256_and_si256(row1[0], bitmask);
  row1[1] = _mm256_and_si256(row1[1], bitmask);
  row1[2] = _mm256_and_si256(row1[2], bitmask);
  row1[3] = _mm256_and_si256(row1[3], bitmask);

  row2[0] = _mm256_and_si256(row2[0], bitmask);
  row2[1] = _mm256_and_si256(row2[1], bitmask);
  row2[2] = _mm256_and_si256(row2[2], bitmask);
  row2[3] = _mm256_and_si256(row2[3], bitmask);

  if (part >= 1)
  {
    row1[4] = _mm256_and_si256(row1[4], bitmask);
    row1[5] = _mm256_and_si256(row1[5], bitmask);

    row2[4] = _mm256_and_si256(row2[4], bitmask);
    row2[5] = _mm256_and_si256(row2[5], bitmask);
  }
}

static uint64_t highds8u_avx2(
    const uint8_t *o,
    int O,
    int xAct,
    int yAct,
    int wAct,
    int hAct)
{
  constexpr int A = (int)sizeof(__m256i);
  const int wActA = ((wAct - xAct) & ~(A - 1)) + xAct;

  __m256i row1[6], row2[6];
  __m256i actSum = _mm256_setzero_si256();
  uint64_t saAct = 0u;
  
  for (int y = yAct; y < hAct; y += 2)
  {
    int x = xAct;
    for (; x < wActA; x += A)
    {
       /**  Row: 0, 1 */
      load_line2<2>(&o[y * O + x], O, row1, row2);
      __m256i c1 = _mm256_add_epi16(row1[0], row2[0]);
      __m256i c2 = _mm256_add_epi16(row1[1], row2[1]);
      __m256i c3 = _mm256_add_epi16(row1[2], row2[2]);
      __m256i c4 = _mm256_add_epi16(row1[3], row2[3]);
      __m256i c5 = _mm256_add_epi16(row1[4], row2[4]);
      __m256i c6 = _mm256_add_epi16(row1[5], row2[5]);

      c1 = _mm256_add_epi16(c1, c6);
      c2 = _mm256_add_epi16(c2, c5);
      c2 = _mm256_add_epi16(_mm256_slli_epi16(c2, 1), c2);
      c3 = _mm256_add_epi16(c3, c4);
      c3 = _mm256_add_epi16(_mm256_slli_epi16(c3, 3), _mm256_slli_epi16(c3, 2));

      __m256i vsum = _mm256_sub_epi16(c3, c1);
      vsum = _mm256_sub_epi16(vsum, c2);

      /**  Row: -1, 2 */
      load_line2<1>(&o[y * O + x], O, row1, row2);
      c1 = _mm256_add_epi16(row1[0], row2[0]);
      c2 = _mm256_add_epi16(row1[1], row2[1]);
      c3 = _mm256_add_epi16(row1[2], row2[2]);
      c4 = _mm256_add_epi16(row1[3], row2[3]);
      c5 = _mm256_add_epi16(row1[4], row2[4]);
      c6 = _mm256_add_epi16(row1[5], row2[5]);

      c1 = _mm256_add_epi16(c1, c6);
      c2 = _mm256_slli_epi16(_mm256_add_epi16(c2, c5), 1);
      c3 = _mm256_add_epi16(c3, c4);
      c3 = _mm256_add_epi16(_mm256_slli_epi16(c3, 1), c3);

      vsum = _mm256_sub_epi16(vsum, c1);
      vsum = _mm256_sub_epi16(vsum, c2);
      vsum = _mm256_sub_epi16(vsum, c3);

      /**  Row: -2, 3 */
      load_line2<0>(&o[y * O + x], O, row1, row2);
      c1 = _mm256_add_epi16(row1[0], row2[0]);
      c2 = _mm256_add_epi16(row1[1], row2[1]);
      c3 = _mm256_add_epi16(row1[2], row2[2]);
      c4 = _mm256_add_epi16(row1[3], row2[3]);

      c1 = _mm256_add_epi16(c1, c4);
      c2 = _mm256_add_epi16(c2, c3);
      
      vsum = _mm256_sub_epi16(vsum, c1);
      vsum = _mm256_sub_epi16(vsum, c2);

      /** Casting into 32bits */
      vsum = _mm256_abs_epi16(vsum);
      vsum = _mm256_hadd_epi16(vsum, vsum);
      actSum = _mm256_add_epi32(actSum, unpack_u16_avx2<0>(vsum));
    }
    for(; x < wAct; x += 2)
    {
      const int f = 12 * ((int)o[y * O + x] + (int)o[y *O + x+1] + (int)o[(y+1)*O + x  ] + (int)o[(y+1)*O + x+1])
                    - 3 * ((int)o[(y-1)*O + x] + (int)o[(y-1)*O + x+1] + (int)o[(y+2)*O + x  ] + (int)o[(y+2)*O + x+1])
                    - 3 * ((int)o[ y   *O + x-1] + (int)o[ y   *O + x+2] + (int)o[(y+1)*O + x-1] + (int)o[(y+1)*O + x+2])
                    - 2 * ((int)o[(y-1)*O + x-1] + (int)o[(y-1)*O + x+2] + (int)o[(y+2)*O + x-1] + (int)o[(y+2)*O + x+2])
                        - ((int)o[(y-2)*O + x-1] + (int)o[(y-2)*O + x  ] + (int)o[(y-2)*O + x+1] + (int)o[(y-2)*O + x+2]
                          + (int)o[(y+3)*O + x-1] + (int)o[(y+3)*O + x  ] + (int)o[(y+3)*O + x+1] + (int)o[(y+3)*O + x+2]
                          + (int)o[(y-1)*O + x-2] + (int)o[ y   *O + x-2] + (int)o[(y+1)*O + x-2] + (int)o[(y+2)*O + x-2]
                          + (int)o[(y-1)*O + x+3] + (int)o[ y   *O + x+3] + (int)o[(y+1)*O + x+3] + (int)o[(y+2)*O + x+3]);
      saAct += (uint64_t) abs(f);
    }
  }

  actSum = _mm256_hadd_epi32(actSum, actSum);
  actSum = _mm256_hadd_epi32(actSum, actSum);
  saAct += _mm256_extract_epi32(actSum, 0);
  saAct += _mm256_extract_epi32(actSum, 4);

  return saAct;
}

static uint64_t highds10u_avx2(
    const uint8_t *src,
    int src_stride,
    int xAct,
    int yAct,
    int wAct,
    int hAct)
{
  constexpr int A = (int)sizeof(__m256i) / 2;
  const int wActA = ((wAct - xAct) & ~(A - 1)) + xAct;

  const uint16_t* o = (const uint16_t*)src;
  const int O = src_stride / 2;

  __m256i row1[6], row2[6];
  __m256i actSum = _mm256_setzero_si256();
  uint64_t saAct = 0u;
  
  for (int y = yAct; y < hAct; y += 2)
  {
    int x;
    for (x = xAct; x < wActA; x += A)
    {
       /**  Row: 0, 1 */
      load_line2<2>(&o[y * O + x], O, row1, row2);
      __m256i c1 = _mm256_add_epi32(row1[0], row2[0]);
      __m256i c2 = _mm256_add_epi32(row1[1], row2[1]);
      __m256i c3 = _mm256_add_epi32(row1[2], row2[2]);
      __m256i c4 = _mm256_add_epi32(row1[3], row2[3]);
      __m256i c5 = _mm256_add_epi32(row1[4], row2[4]);
      __m256i c6 = _mm256_add_epi32(row1[5], row2[5]);

      c1 = _mm256_add_epi32(c1, c6);
      c2 = _mm256_add_epi32(c2, c5);
      c2 = _mm256_add_epi32(_mm256_slli_epi32(c2, 1), c2);
      c3 = _mm256_add_epi32(c3, c4);
      c3 = _mm256_add_epi32(_mm256_slli_epi32(c3, 3), _mm256_slli_epi32(c3, 2));

      __m256i vsum = _mm256_sub_epi32(c3, c1);
      vsum = _mm256_sub_epi32(vsum, c2);

      /**  Row: -1, 2 */
      load_line2<1>(&o[y * O + x], O, row1, row2);
      c1 = _mm256_add_epi32(row1[0], row2[0]);
      c2 = _mm256_add_epi32(row1[1], row2[1]);
      c3 = _mm256_add_epi32(row1[2], row2[2]);
      c4 = _mm256_add_epi32(row1[3], row2[3]);
      c5 = _mm256_add_epi32(row1[4], row2[4]);
      c6 = _mm256_add_epi32(row1[5], row2[5]);

      c1 = _mm256_add_epi32(c1, c6);
      c2 = _mm256_slli_epi32(_mm256_add_epi32(c2, c5), 1);
      c3 = _mm256_add_epi32(c3, c4);
      c3 = _mm256_add_epi32(_mm256_slli_epi32(c3, 1), c3);

      vsum = _mm256_sub_epi32(vsum, c1);
      vsum = _mm256_sub_epi32(vsum, c2);
      vsum = _mm256_sub_epi32(vsum, c3);

      /**  Row: -2, 3 */
      load_line2<0>(&o[y * O + x], O, row1, row2);
      c1 = _mm256_add_epi32(row1[0], row2[0]);
      c2 = _mm256_add_epi32(row1[1], row2[1]);
      c3 = _mm256_add_epi32(row1[2], row2[2]);
      c4 = _mm256_add_epi32(row1[3], row2[3]);

      c1 = _mm256_add_epi32(c1, c4);
      c2 = _mm256_add_epi32(c2, c3);
      
      vsum = _mm256_sub_epi32(vsum, c1);
      vsum = _mm256_sub_epi32(vsum, c2);

      /** Casting into 32bits */
      vsum = _mm256_abs_epi32(vsum);
      actSum = _mm256_add_epi32(actSum, vsum);
    }
    for(; x < wAct; x += 2)
    {
      const int f = 12 * ((int)o[y * O + x] + (int)o[y *O + x+1] + (int)o[(y+1)*O + x  ] + (int)o[(y+1)*O + x+1])
                    - 3 * ((int)o[(y-1)*O + x] + (int)o[(y-1)*O + x+1] + (int)o[(y+2)*O + x  ] + (int)o[(y+2)*O + x+1])
                    - 3 * ((int)o[ y   *O + x-1] + (int)o[ y   *O + x+2] + (int)o[(y+1)*O + x-1] + (int)o[(y+1)*O + x+2])
                    - 2 * ((int)o[(y-1)*O + x-1] + (int)o[(y-1)*O + x+2] + (int)o[(y+2)*O + x-1] + (int)o[(y+2)*O + x+2])
                        - ((int)o[(y-2)*O + x-1] + (int)o[(y-2)*O + x  ] + (int)o[(y-2)*O + x+1] + (int)o[(y-2)*O + x+2]
                          + (int)o[(y+3)*O + x-1] + (int)o[(y+3)*O + x  ] + (int)o[(y+3)*O + x+1] + (int)o[(y+3)*O + x+2]
                          + (int)o[(y-1)*O + x-2] + (int)o[ y   *O + x-2] + (int)o[(y+1)*O + x-2] + (int)o[(y+2)*O + x-2]
                          + (int)o[(y-1)*O + x+3] + (int)o[ y   *O + x+3] + (int)o[(y+1)*O + x+3] + (int)o[(y+2)*O + x+3]);
      saAct += (uint64_t) abs(f);
    }
  }

  actSum = _mm256_hadd_epi32(actSum, actSum);
  actSum = _mm256_hadd_epi32(actSum, actSum);
  saAct += _mm256_extract_epi32(actSum, 0);
  saAct += _mm256_extract_epi32(actSum, 4);

  return saAct;
}

spatial_act_func_t get_spatial_act_avx2_func(int bit_depth, bool down_sampling)
{
  constexpr spatial_act_func_t FUNC[] = {high8u_avx2, high10u_avx2, highds8u_avx2, highds10u_avx2};

  int idx = (bit_depth > 8 ? 1 : 0) + (down_sampling ? 2 : 0);
  return FUNC[idx];
}