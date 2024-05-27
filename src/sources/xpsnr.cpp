#include "xpsnr.hpp"
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>

#include "ta_act.h"
#include "sa_act.h"
#include "distortion.h"

static xpsnr_cpu_t detect_cpu_support()
{
#ifdef __GNUC__
#ifdef __AVX2__
  return xpsnr_cpu_avx2;
#elif defined(__SSE4_1__)
  return xpsnr_cpu_sse41;
#endif
#elif defined(_MSC_VER) && (_MSC_VER >= 1800)
  int CPUInfo[4] = {0};
  __cpuid(CPUInfo, 0);
  int nIds = CPUInfo[0];
  if (nIds >= 7)
  {
    __cpuidex(CPUInfo, 7, 0);
    if (CPUInfo[1] & (1 << 5))
    {
      return xpsnr_cpu_avx2;
    }
    if (CPUInfo[1] & (1 << 19))
    {
      return xpsnr_cpu_sse41;
    }
  }
#endif

  return xpsnr_cpu_c;
}

int xpsnr::init(const xpsnr_options_t &opt)
{
  cpu = detect_cpu_support();
  if (opt.cpu != xspnr_cpu_auto)
  {
    cpu = (xpsnr_cpu_t)std::min((int)cpu, (int)opt.cpu);
  }

  W = opt.width;
  H = opt.height;
  BD = opt.bit_depth;
  fps = (int)std::round(opt.frame_rate);

  const double r = double(W * H) / (3840.0 * 2160.0);
  const int linesize = W * (BD > 8 ? 2 : 1);

  N = std::max(0, 4 * (int)(32.0 * std::sqrt(r) + 0.5));
  nblk_w = (W + N - 1) / N;
  nblk_h = (H + N - 1) / N;

  act_pic = std::sqrt(16.0 * (1 << (2 * BD - 9)) / std::sqrt(std::max(r, 0.00001)));
  act_min = (double)(1 << (BD - 6));

  for (int i = 0; i < 3; ++i)
  {
    buf.luma_pel[i].reset((uint8_t *)aligned_alloc(64, linesize * H), &free);
  }

  buf.luma_sse.resize(nblk_w * nblk_h);
  buf.weights.resize(nblk_w * nblk_h);

  input_counts = 0;

  return 0;
}

int xpsnr::put_frame(xpsnr_putframe_exchanges_t &pe)
{
  const bool ds = (W * H > 2048 * 1152); /* This is little bit different from JITU paper (2048 * 1280) */
  const int order = (input_counts >= 2 && fps > 32) ? 2 : (input_counts >= 1 ? 1 : 0);
  const ssd_func_t ssd_func = get_ssd_func(cpu, BD);
  const spatial_act_func_t sa_func = get_saact_func(cpu, BD, ds);
  const temp_act_func_t ta_func = (order > 0) ? get_temp_act_func(cpu, BD, order, ds) : nullptr;

  // Copy Luma only
  uint8_t *o = buf.luma_pel[input_counts % 3].get();
  uint8_t *oM1 = buf.luma_pel[(input_counts + 2) % 3].get();
  uint8_t *oM2 = buf.luma_pel[(input_counts + 1) % 3].get();
  const uint32_t stride = W * (BD > 8 ? 2 : 1);

  std::memcpy(o, pe.ref_ptr, stride * H);

  // Compute Weights and SSE
  const uint8_t *r = (const uint8_t *)pe.dist_ptr;
  const int SHIFT = BD > 8 ? 1 : 0;

  if (N >= 4)
  {
    const bool blockWeightSmoothing = (W * H) <= (640 * 480);
    const int bVal = ds ? 2 : 1;

    for (int y = 0, bidx = 0; y < H; y += N)
    {
      int blockHeight = (y + N > H ? H - y : N);

      for (int x = 0; x < W; x += N, ++bidx)
      {
        int blockWidth = (x + N > W ? W - x : N);
        int xAct = (x > 0 ? 0 : bVal);
        int yAct = (y > 0 ? 0 : bVal);
        int wAct = (x + blockWidth < W ? blockWidth : blockWidth - bVal);
        int hAct = (y + blockHeight < H ? blockHeight : blockHeight - bVal);

        uint64_t sse = ssd_func(&o[y * stride + (x << SHIFT)], stride,
                                &r[y * stride + (x << SHIFT)], stride,
                                blockWidth, blockHeight);

        uint64_t sa_act = sa_func(&o[y * stride + (x << SHIFT)], stride,
                                  xAct, yAct, wAct, hAct);

        uint64_t ta_act = 0;
        if (ta_func)
        {
          ta_act = ta_func(&o[y * stride + (x << SHIFT)],
                           &oM1[y * stride + (x << SHIFT)],
                           &oM2[y * stride + (x << SHIFT)],
                           stride, wAct, hAct);
        }

        double act = ((double)sa_act / (wAct * hAct));
        if (ta_act)
        {
          act += XPSNR_GAMMA * ((double)ta_act / (blockWidth * blockHeight));
        }

        act = std::max(act_min, act);
        double weight = act_pic / act;

        buf.weights[bidx] = weight;
        buf.luma_sse[bidx] = sse;

        if (blockWeightSmoothing) /* inline "minimum-smoothing" as in paper */
        {
          double msActPrev = 0;
          if (x == 0) /* first column */
          {
            msActPrev = (bidx > 1 ? buf.weights[bidx - 2] : 0);
          }
          else /* after first column */
          {
            msActPrev = (x > N ? std::max(buf.weights[bidx - 2], buf.weights[bidx]) : buf.weights[bidx]);
          }
          if (bidx > nblk_w) /* after first row and first column */
          {
            msActPrev = std::max(msActPrev, buf.weights[bidx - 1 - nblk_w]); /* min (left, top) */
          }
          if ((bidx > 0) && (buf.weights[bidx - 1] > msActPrev))
          {
            buf.weights[bidx - 1] = msActPrev;
          }
          if ((x + N >= W) && (y + N >= H) && (bidx > nblk_w)) /* last block in picture */
          {
            msActPrev = std::max(buf.weights[bidx - 1], buf.weights[bidx - nblk_w]);
            if (buf.weights[bidx] > msActPrev)
            {
              buf.weights[bidx] = msActPrev;
            }
          }
        }
      }
    }

    double wsse_luma = 0;
    for (int i = 0; i < nblk_w * nblk_h; ++i)
    {
      wsse_luma += buf.weights[i] * buf.luma_sse[i];
    }

    pe.xpsnr = compute_wpsnr_frame(wsse_luma);
  }

  ++input_counts;
  return 0;
}

double xpsnr::compute_wpsnr_frame(const double wsse_sum)
{
  return 10.0 * std::log10(
                    (double)((int64_t)W * H * (1 << (BD - 1)) * (1 << (BD - 1))) / wsse_sum);
}