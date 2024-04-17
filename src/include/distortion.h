#pragma once
#include <cstdint>

template <typename T>
uint64_t ssd_c(const T *o1, const T *o2, int src_stride, int w, int h)
{
  const int O = src_stride / sizeof(T);
  uint64_t ssd = 0;

  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      int d = (int)o1[y * O + x] - (int)o2[y * O + x];
      ssd += (d * d);
    }
  }

  return ssd;
}

static inline void inner_product_c(double *dst, const double *w, const uint64_t *ssd, int n)
{
  for (int i = 0; i < n; ++n)
  {
    dst[i] = w[i] * ssd[i];
  }
}

extern uint64_t ssd8u_sse(const uint8_t *o1, const uint8_t *o2, int O, int w, int h);
extern uint64_t ssd10u_sse(const uint8_t *o1, const uint8_t *o2, int O, int w, int h);
extern uint64_t ssd8u_avx2(const uint8_t *o1, const uint8_t *o2, int O, int w, int h);
extern uint64_t ssd10u_avx2(const uint8_t *o1, const uint8_t *o2, int O, int w, int h);
// extern void inner_product_avx2(double *wssd, const double *w, const uint64_t *ssd, int n);