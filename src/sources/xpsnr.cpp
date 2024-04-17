#include "xpsnr.h"
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>

#include "ta_act.h"
#include "sa_act.h"
#include "distortion.h"

struct xpsnr_context_t
{
  xpsnr_cpu_t cpu;
  int W;
  int H;
  int N;
  int BD;
  int NW;
  int NH;
  int ifps;
  double act_min;
  double act_pic_sqrt;
  bool use_downsampling;
  bool use_weight_smoothing;
};

static int get_block_size(int width, int height)
{
  // Decompose 128 into 4 and 32 to ensure alignment with SIMD memory requirements.
  double r = (double)(width * height) / (3840.0 * 2160.0);
  return std::max(0, 4 * (int)(32.0 * std::sqrt(r) + 0.5));
}

int get_xpsnr_structures(
    const int luma_width,
    const int luma_height,
    const double frame_rate,
    int &block_size,
    int &nblks_in_width,
    int &nblks_in_height)
{
  int N = get_block_size(luma_width, luma_height);

  block_size = N;
  nblks_in_width = (luma_width + N - 1) / N;
  nblks_in_height = (luma_height + N - 1) / N;

  return 0;
}

static xpsnr_cpu_t detect_cpu_support()
{
  if (__builtin_cpu_supports("avx2"))
  {
    return xpsnr_cpu_avx2;
  }
  else if (__builtin_cpu_supports("sse4.1"))
  {
    return xpsnr_cpu_sse41;
  }

  return xpsnr_cpu_c;
}

static int init_xpsnr_context(
    xpsnr_context_t &ctx,
    const int luma_width,
    const int luma_height,
    const int bit_depth,
    const double frame_rate,
    const xpsnr_cpu_t cpu)
{
  get_xpsnr_structures(luma_width, luma_height, frame_rate, ctx.N, ctx.NW, ctx.NH);

  ctx.cpu = cpu == xspnr_cpu_auto ? detect_cpu_support() : cpu;
  ctx.W = luma_width;
  ctx.H = luma_height;
  ctx.BD = bit_depth;
  ctx.ifps = (int)round(frame_rate);

  const double R = (double)(luma_width * luma_height) / (3840.0 * 2160.0);

  ctx.act_min = (double)(1 << (bit_depth - 6));
  ctx.act_pic_sqrt = std::sqrt((double)(1 << (2 * bit_depth - 9)) / std::sqrt(R));
  ctx.use_downsampling = (luma_width * luma_height > 2048 * 1152); /* This is little bit different from JITU (2048 * 1280) */
  ctx.use_weight_smoothing = (luma_width * luma_height <= 640 * 480);

  return 0;
}

static int calc_wssd_in_picture(
    const xpsnr_context_t &ctx,
    const void *org_ptr,
    const void *dist_ptr,
    int stride,
    const double *weights,
    double *wssd_out,
    uint64_t *ssd_out)
{
  const decltype(&ssd8u_sse) fn_ssd = ctx.BD > 10 ? ssd10u_avx2 : ssd8u_avx2;
  auto o = (const uint8_t *)org_ptr;
  auto d = (const uint8_t *)dist_ptr;
  const int M = ctx.BD > 10 ? 2 : 1;

  int blk_idx = 0;

  for (int y = 0; y < ctx.H; y += ctx.N)
  {
    for (int x = 0; x < ctx.W; x += ctx.N, ++blk_idx)
    {
      const int BH = std::min(ctx.H, y + ctx.N) - y;
      const int BW = std::min(ctx.W, x + ctx.N) - x;

      ssd_out[blk_idx] = fn_ssd(&o[(y * stride + x) * M], &d[(y * stride + x) * M], stride, BW, BH);
    }
  }

  inner_product_c(wssd_out, weights, ssd_out, ctx.NW * ctx.NH);

  return 0;
}

static int calc_weights_in_picture(
    const xpsnr_context_t &ctx,
    const void *org_ptr,
    const void *prev_ptr,
    const void *pprev_ptr,
    int stride,
    double *weights)
{
  auto picOrg = (const uint8_t*)org_ptr;
  auto picOrgM1 = (const uint8_t*)prev_ptr;
  auto picOrgM2 = (const uint8_t*)pprev_ptr;
  const int PAD = ctx.use_downsampling ? 2 : 1;
  const int BPP = (ctx.BD > 8 ? 2 : 1);

  spatial_act_func_t sa_func = nullptr;
  temp_act_func_t ta_func = nullptr;

  int ta_order = (pprev_ptr && ctx.ifps > 32) ? 2 : (prev_ptr ? 1 : 0);

  switch (ctx.cpu)
  {
  case xpsnr_cpu_c:
    sa_func = get_spatial_act_func(ctx.BD, ctx.use_downsampling);
    break;
  case xpsnr_cpu_sse41:
    sa_func = get_spatial_act_sse_func(ctx.BD, ctx.use_downsampling);
    break;
  case xpsnr_cpu_avx2:
    sa_func = get_spatial_act_avx2_func(ctx.BD, ctx.use_downsampling);
    break;
  default:
    break;
  }
  if (ta_order)
  {
    switch (ctx.cpu)
    {
    case xpsnr_cpu_c:
      ta_func = get_temp_act_func(ctx.BD, ta_order, ctx.use_downsampling);
      break;
    case xpsnr_cpu_sse41:
      ta_func = get_temp_act_sse_func(ctx.BD, ta_order, ctx.use_downsampling);
      break;
    case xpsnr_cpu_avx2:
      ta_func = get_temp_act_avx2_func(ctx.BD, ta_order, ctx.use_downsampling);
      break;
    default:
      break;
    }
  }

  double act_prev = 0.0;
  int blk_idx = 0;

  for (int y = 0; y < ctx.H; y += ctx.N)
  {
    for (int x = 0; x < ctx.W; x += ctx.N, ++blk_idx)
    {
      const int BH = std::min(ctx.H, y + ctx.N) - y;
      const int BW = std::min(ctx.W, x + ctx.N) - x;

      const uint8_t *o = &picOrg[(y * ctx.W + x) * BPP];
      const uint8_t *oM1 = &picOrgM1[(y * ctx.W + x) * BPP];
      const uint8_t *oM2 = &picOrgM2[(y * ctx.W + x) * BPP];
      const int xAct = x > 0 ? 0 : PAD;
      const int yAct = y > 0 ? 0 : PAD;
      const int wAct = (x + BW < ctx.W ? BW : BW - PAD);
      const int hAct = (y + BH < ctx.H ? BH : BH - PAD);

      double act = ctx.act_min;

      if (wAct > xAct && hAct > yAct)
      {
        uint64_t sa_act = sa_func(o, ctx.W * BPP, xAct, yAct, wAct, hAct);
        act = (double)sa_act / ((double)(wAct - xAct) * (double)(hAct - yAct));

        if (ta_func)
        {
          uint64_t ta_act = ta_func(o, oM1, oM2, ctx.W * BPP, BW, BH);
          act += (double)ta_act / ((double)BW * (double)BH);
        }
        act /= 4.0;
        act = std::max(act, ctx.act_min);
      }

      double w = ctx.act_pic_sqrt / act;
      weights[blk_idx] = w;

      // NOTE: I didn't checked this code. so there is a possibility that this code is wrong.
      if (ctx.use_weight_smoothing)
      {
        if (x == 0) /* first column */
        {
          act_prev = (blk_idx > 1 ? weights[blk_idx - 2] : 0);
        }
        else /* after first column */
        {
          act_prev = (x > ctx.N ? std::max(weights[blk_idx - 2], weights[blk_idx]) : weights[blk_idx]);
        }
        if (blk_idx > ctx.NW) /* after first row and first column */
        {
          act_prev = std::max(act_prev, weights[blk_idx - 1 - ctx.NW]); /* min (left, top) */
        }
        if ((blk_idx > 0) && (weights[blk_idx - 1] > act_prev))
        {
          weights[blk_idx - 1] = act_prev;
        }
        if ((x + ctx.N >= ctx.W) && (y + ctx.N >= ctx.H) && (blk_idx > ctx.NW)) /* last block in picture */
        {
          act_prev = std::max(weights[blk_idx - 1], weights[blk_idx - ctx.NW]);
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

int calc_xpsnr_in_picture(
    const int luma_width,
    const int luma_height,
    const int bit_depth,
    const double frame_rate,
    const void *curr_yuv_ptr,
    const void *dist_yuv_ptr,
    const void *prev_yuv_ptr,
    const void *pprev_yuv_ptr,
    const int yuv_stride,
    const xpsnr_cpu_t cpu,
    double *weights_out,
    uint64_t *ssd_out,
    double *wssd_out)
{
  assert(curr_yuv_ptr && weights_out);

  xpsnr_context_t ctx = {};

  init_xpsnr_context(ctx, luma_width, luma_height, bit_depth, frame_rate, cpu);

  calc_weights_in_picture(ctx, curr_yuv_ptr, prev_yuv_ptr, pprev_yuv_ptr, yuv_stride, weights_out);

  if (!ssd_out || !wssd_out)
  {
    return 0;
  }

  calc_wssd_in_picture(ctx, curr_yuv_ptr, dist_yuv_ptr, yuv_stride, weights_out, wssd_out, ssd_out);

  return 0;
}

