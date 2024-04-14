#include "xpsnr.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <iostream>
#include <limits>

static int get_block_size(const int width, const int height, const int bit_depth)
{
  // Decompose 128 into 4 and 32 to ensure alignment with SIMD memory requirements.
  double r = (double)(width * height) / (3840.0 * 2160.0);
  return std::max(0, 4 * (int)(32.0 * std::sqrt(r) + 0.5));
}

static inline void frame_copy(
    uint8_t *dst,
    const uint8_t *src,
    int srcStride,
    int width,
    int height,
    int BPP)
{
  if (width * BPP == srcStride)
  {
    std::memcpy(dst, src, width * height * BPP);
  }
  else
  {
    const int dst_stride = width * BPP;
    for (int i = 0; i < height; ++i)
    {
      std::memcpy(dst, src, width * BPP);
      dst += dst_stride;
      src += srcStride;
    }
  }
}

int xpsnr::init(const xpsnr_init_params &iparam)
{
  clear();

  // Constants
  W = iparam.pic_width;
  H = iparam.pic_height;
  N = get_block_size(W, H, iparam.bit_depth);
  ifps = (int)std::round(iparam.fps);
  BD = iparam.bit_depth > 8 ? 10 : 8;
  NW = (W + N - 1) / N;
  NH = (H + N - 1) / N;

  const double R = (double)(W * H) / (3840.0 * 2160.0);
  act_min = (double)(1 << (BD - 6));
  act_pic_sqrt = std::sqrt((double)(1 << (2 * BD - 9)) / std::sqrt(R));
  use_downsampling = (W * H > 2048 * 1152); /* This is little bit different from JITU (2048 * 1280) */
  use_weight_smoothing = (W * H <= 640 * 480);

  next_index = 0;
  next_output_index = 0;
  buffer_len_max = std::max(iparam.ring_buffer_len, 3);

  orgpic_array.resize(buffer_len_max);
  weight_array.resize(buffer_len_max);
  for (int i = 0; i < buffer_len_max; ++i)
  {
    orgpic_array[i].resize(W * H * (BD > 8 ? 2 : 1));
    weight_array[i].resize(NW * NH);
  }

  return 0;
}

int xpsnr::clear()
{
  orgpic_array.clear();
  weight_array.clear();
  buffer_len_max = 0;
  return 0;
}

int xpsnr::put_frame(const uint8_t *input_yuv,
                     int yuv_stride,
                     int64_t index)
{
  if (index != next_index)
  {
    std::cout << "[xpsnr] input index is not sequential: expected=" << next_index << ", incomed=" << index << std::endl;
    std::cout.flush();
    return -1;
  }

  const int bfridx = (int)(next_index % buffer_len_max);

  frame_copy(orgpic_array[bfridx].data(), input_yuv, yuv_stride, W, H, BD > 8 ? 2 : 1);
  ++next_index;

  return 0;
}

double xpsnr::get_xpsnr_sync(int64_t index)
{
  if (index >= next_index ||
      index <= (next_index - buffer_len_max))
  {
    std::cout << "[xpsnr] " << index << " xpsnr value can't accessable." << std::endl;
    std::cout.flush();

    return std::numeric_limits<double>::quiet_NaN();
  }
  return 0.0;
}

int xpsnr::calc_weights_in_pic(int64_t index)
{
  const int bfridx = (int)(index % buffer_len_max);
  const int bfridx_m1 = (int)((index - 1 + buffer_len_max) % buffer_len_max);
  const int bfridx_m2 = (int)((index - 2 + buffer_len_max) % buffer_len_max);
  const uint8_t *picOrg = orgpic_array[bfridx].data();
  const uint8_t *picOrgM1 = orgpic_array[bfridx_m1].data();
  const uint8_t *picOrgM2 = orgpic_array[bfridx_m2].data();
  double *weights = weight_array[bfridx].data();
  const int PAD = use_downsampling ? 2 : 1;
  const int BPP = (BD > 8 ? 2 : 1);

  spatial_act_func_t sa_func = nullptr;
  temp_act_func_t ta_func = nullptr;

  sa_func = get_spatial_act_avx2_func(BD, use_downsampling);

  if (index > 1 && ifps > 32)
  {
    ta_func = get_temp_act_avx2_func(BD, 2, use_downsampling);
  }
  else if (index > 0)
  {
    ta_func = get_temp_act_avx2_func(BD, 1, use_downsampling);
  }

  double act_prev = 0.0;
  int blk_idx = 0;

  for (int y = 0; y < H; y += N)
  {
    for (int x = 0; x < W; x += N, ++blk_idx)
    {
      const int BH = std::min(H, y + N) - y;
      const int BW = std::min(W, x + N) - x;

      const uint8_t *o = &picOrg[(y * W + x) * BPP];
      const uint8_t *oM1 = &picOrgM1[(y * W + x) * BPP];
      const uint8_t *oM2 = &picOrgM2[(y * W + x) * BPP];
      const int xAct = x > 0 ? 0 : PAD;
      const int yAct = y > 0 ? 0 : PAD;
      const int wAct = (x + BW < W ? BW : BW - PAD);
      const int hAct = (y + BH < H ? BH : BH - PAD);

      double act = act_min;

      if (wAct > xAct && hAct > yAct)
      {
        uint64_t sa_act = sa_func(o, W * BPP, xAct, yAct, wAct, hAct);
        act = (double)sa_act / ((double)(wAct - xAct) * (double)(hAct - yAct));

        if (ta_func)
        {
          uint64_t ta_act = ta_func(o, oM1, oM2, W * BPP, BW, BH);
          act += (double)ta_act / ((double)BW * (double)BH);
        }
        act /= 4.0;
        act = std::max(act, act_min);
      }

      double w = act_pic_sqrt / act;
      weights[blk_idx] = w;

      // NOTE: I didn't checked this code. so there is a possibility that this code is wrong.
      if (use_weight_smoothing)
      {
        if (x == 0) /* first column */
        {
          act_prev = (blk_idx > 1 ? weights[blk_idx - 2] : 0);
        }
        else /* after first column */
        {
          act_prev = (x > N ? std::max(weights[blk_idx - 2], weights[blk_idx]) : weights[blk_idx]);
        }
        if (blk_idx > NW) /* after first row and first column */
        {
          act_prev = std::max(act_prev, weights[blk_idx - 1 - NW]); /* min (left, top) */
        }
        if ((blk_idx > 0) && (weights[blk_idx - 1] > act_prev))
        {
          weights[blk_idx - 1] = act_prev;
        }
        if ((x + N >= W) && (y + N >= H) && (blk_idx > NW)) /* last block in picture */
        {
          act_prev = std::max(weights[blk_idx - 1], weights[blk_idx - NW]);
          if (weights[blk_idx] > act_prev)
          {
            weights[blk_idx] = act_prev;
          }
        }
      }
    }
  }

  return 0;
}