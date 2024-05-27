#pragma once
#include "xpsnr.hpp"

typedef uint64_t (*ssd_func_t)(
    const uint8_t *ref_ptr,
    const int ref_stride,
    const uint8_t *tar_ptr,
    const int tar_stride,
    const int w,
    const int h);

extern ssd_func_t get_ssd_func(
    const xpsnr_cpu_t cpu,
    int bit_depth);
