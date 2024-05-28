#pragma once
#include <cstdint>
#include <vector>
#include <memory>

enum xpsnr_cpu_t : int
{
  xspnr_cpu_auto = 0,
  xpsnr_cpu_c,
  xpsnr_cpu_sse41,
  xpsnr_cpu_avx2
};

struct xpsnr_options_t
{
  int width;
  int height;
  int bit_depth;
  double frame_rate;
  xpsnr_cpu_t cpu;
};

struct xpsnr_putframe_exchanges_t
{
  const void *ref_ptr;  // [in] pointer of reference yuv signal
  const void *dist_ptr; // [in] pointer of distorted yuv signal
  double xpsnr;         // [out] XPSNR score of this frame
};

class xpsnr
{
public:
  int init(const xpsnr_options_t &opt);
  int put_frame(xpsnr_putframe_exchanges_t &pe);

private:
  double compute_wpsnr_frame(const double wsse_sum);

private:
  xpsnr_cpu_t cpu;
  int W, H;           // picture luma width/height
  int BD;             // bit depth
  int fps;            // frame rate
  int N;              // block size
  int nblk_w, nblk_h; // number of blocks in width/height
  double avg_act;     // equivalents to sqrt(16 * average overall activity)
  double act_min;     // minimum activity

  struct
  {
    std::shared_ptr<uint8_t> luma_pel[3];
    std::vector<uint64_t> luma_sse;
    std::vector<double> weights;
  } buf;

  int64_t input_counts;
};