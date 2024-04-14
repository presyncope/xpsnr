#pragma once
#include <cstdint>

static constexpr int XPSNR_GAMMA = 2;

typedef uint64_t (*temp_act_func_t)(
    const uint8_t *src,
    const uint8_t *src_m1,
    const uint8_t *src_m2,
    int src_stride,
    int wAct,
    int hAct);

extern temp_act_func_t get_temp_act_func(
    const int bit_depth,
    const int diff_order, // 1 or 2
    bool down_sampling);

extern temp_act_func_t get_temp_act_sse_func(
    const int bit_depth,
    const int diff_order, // 1 or 2
    bool down_sampling);

extern temp_act_func_t get_temp_act_avx2_func(
    const int bit_depth,
    const int diff_order, // 1 or 2
    bool down_sampling);