#include "ta_act.h"
#include <algorithm>

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

temp_act_func_t get_temp_act_func(
    const int bit_depth,
    const int diff_order, // 1 or 2
    bool down_sampling)
{
  // [ds][order][bd]
  constexpr temp_act_func_t func_table[2 * 2 * 2] = {
      diff_impl_c<uint8_t, 1, false>, diff_impl_c<uint16_t, 1, false>,
      diff_impl_c<uint8_t, 2, false>, diff_impl_c<uint16_t, 2, false>,
      diff_impl_c<uint8_t, 1, true>, diff_impl_c<uint16_t, 1, true>,
      diff_impl_c<uint8_t, 2, true>, diff_impl_c<uint16_t, 2, true>};

  int idx = std::max(std::min((bit_depth + 7) / 8, 2), 1) - 1;
  idx += (std::max(std::min(diff_order, 2), 1) - 1) * 2;
  idx += (down_sampling ? 1 : 0) * 4;

  return func_table[idx];
}