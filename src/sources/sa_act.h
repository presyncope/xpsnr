#pragma once
#include "xpsnr.hpp"

typedef uint64_t (*spatial_act_func_t)(
    const uint8_t *src,
    int src_stride,
    int xAct,
    int yAct,
    int wAct,
    int hAct);

extern spatial_act_func_t get_saact_func(xpsnr_cpu_t cpu, int bit_depth, bool down_sampling);
