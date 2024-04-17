#pragma once
#include <cstdint>

enum xpsnr_cpu_t
{
  xspnr_cpu_auto = 0,
  xpsnr_cpu_c,
  xpsnr_cpu_sse41,
  xpsnr_cpu_avx2
};

extern int get_xpsnr_structures(
    const int luma_width,
    const int luma_height,
    const double frame_rate,
    int &block_size,
    int &nblks_in_width,
    int &nblks_in_height);

extern int calc_xpsnr_in_picture(
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
    double *wssd_out);