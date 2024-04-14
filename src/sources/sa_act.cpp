#include "sa_act.h"
#include <cmath>

template <typename T, bool ds>
uint64_t spatial_act_impl_c(
    const uint8_t *src,
    int src_stride,
    int xAct,
    int yAct,
    int wAct,
    int hAct)
{
  const T *o = (const T *)src;
  const int O = src_stride / sizeof(T);

  uint64_t saAct = 0;

  if(!ds)
  {
    for (int y = yAct; y < hAct; ++y)
    {
      for (int x = xAct; x < wAct; ++x)
      {
        const int f = 12 * (int)o[y * O + x] 
          - 2 * ( (int)o[y * O + x - 1] + (int)o[y * O + x + 1] + (int)o[(y - 1) * O + x] + (int)o[(y + 1) * O + x] ) 
          - ( (int)o[(y - 1) * O + x - 1] + (int)o[(y - 1) * O + x + 1] + (int)o[(y + 1) * O + x - 1] + (int)o[(y + 1) * O + x + 1]);
        saAct += (uint64_t)std::abs(f);
      }
    }
  }
  else
  {
    for (int y = yAct; y < hAct; y += 2)
    {
      for (int x = xAct; x < wAct; x += 2)
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
  }

  return saAct;
}

spatial_act_func_t get_spatial_act_func(int bit_depth, bool down_sampling)
{
  constexpr spatial_act_func_t FUNC[] = {
      spatial_act_impl_c<uint8_t, false>, spatial_act_impl_c<uint16_t, false>,
      spatial_act_impl_c<uint8_t, true>, spatial_act_impl_c<uint16_t, true>};

  int idx = (bit_depth > 8 ? 1 : 0) + (down_sampling ? 2 : 0);
  return FUNC[idx];
}