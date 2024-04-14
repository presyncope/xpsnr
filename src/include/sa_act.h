#pragma once
#include <cstdint>

typedef uint64_t (*spatial_act_func_t)(
    const uint8_t *src,
    int src_stride,
    int xAct,
    int yAct,
    int wAct,
    int hAct);

extern spatial_act_func_t get_spatial_act_func(int bit_depth, bool down_sampling);
extern spatial_act_func_t get_spatial_act_sse_func(int bit_depth, bool down_sampling);
extern spatial_act_func_t get_spatial_act_avx2_func(int bit_depth, bool down_sampling);
