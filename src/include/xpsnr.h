#pragma once
#include <vector>
#include <cstdint>
#include "sa_act.h"
#include "ta_act.h"

struct xpsnr_init_params
{
  int pic_width;
  int pic_height;
  int bit_depth;   // 8 or 10
  double fps;
  int ring_buffer_len;
};

class xpsnr
{
public:
  int init(const xpsnr_init_params &iparam);

  int clear();

  int put_frame(const uint8_t *input_yuv,
                int yuv_stride,
                int64_t index);

  double get_xpsnr_sync(int64_t index);

  /** Private varaibles */
private:
  int W, H;                  /* Picture luma width and height*/
  int N;                     /* Block size */
  int ifps;                  /* Integer value of frame rate */
  int BD;                    /* Bit depth*/
  double act_min;            /* minimum allowed value of visual activity */
  double act_pic_sqrt;       /* Squared root value of average overall activity */
  bool use_downsampling;     /* enable downsampling if true */
  bool use_weight_smoothing; /* enable weight smoothing if true */
  int NW, NH;                /* Number of blocks in width/height */
  int64_t next_index;        /* Next input picture's POC */
  int64_t next_output_index;
  int buffer_len_max;
  std::vector<std::vector<uint8_t>> orgpic_array;
  std::vector<std::vector<double>> weight_array;

  /** Private functions*/
private:
  int calc_weights_in_pic(int64_t index);
};
