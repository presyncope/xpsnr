#include "sa_act.h"

#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>

struct sa_test_case_t
{
  int width;
  int height;
  std::vector<uint8_t> samples8u;
  std::vector<uint16_t> samples10u;

  void resize(size_t size)
  {
    samples8u.resize(size);
    samples10u.resize(size);
  }

  int get_block_size() const
  {
    double r = (double)(width * height) / (3840.0 * 2160.0);
    return std::max(0, 4 * (int)(32.0 * std::sqrt(r) + 0.5));
  }
};

class SaActTest: public testing::Test
{
protected:
  std::vector<sa_test_case_t> testset;
  
  void SetUp() override
  {
    std::mt19937 gen{1313};
    std::uniform_int_distribution<uint8_t> dis8u(0, 255);
    std::uniform_int_distribution<uint16_t> dis10u(0, 1023);

    testset.clear();

    constexpr int resolutions[6][2]{
        {3840, 2160},
        {1920, 1080},
        {1440, 960},
        {1280, 720},
        {1056, 592},
        {1024, 576},
    };

    for (int ridx = 0; ridx < 6; ++ridx)
    {
      int w = resolutions[ridx][0];
      int h = resolutions[ridx][1];

      for (int i = 0; i < 2; ++i)
      {
        sa_test_case_t tc{};
        tc.width = w;
        tc.height = h;
        tc.resize(w * h);

        for (int j = 0; j < w * h; ++j)
        {
          tc.samples8u[j] = dis8u(gen);
          tc.samples10u[j] = dis10u(gen);
        }

        testset.emplace_back(std::forward<decltype(tc)>(tc));
      }
    }
  }
};

TEST_F(SaActTest, SaAct8u)
{
  auto sa8u_c = get_saact_func(xpsnr_cpu_c, 8, false);
  auto sa8u_sse = get_saact_func(xpsnr_cpu_sse41, 8, false);
  auto sa8u_avx2 = get_saact_func(xpsnr_cpu_avx2, 8, false);
  constexpr int bVal = 1;

  for (auto &tc : testset)
  {
    const int W = tc.width;
    const int H = tc.height;
    const int B = tc.get_block_size();

    for (int y = 0; y < H; y += B)
    {
      for (int x = 0; x < W; x += B)
      {
        const int O = tc.width;
        const uint8_t* o = tc.samples8u.data() + y * O + x;
        const int blockWidth = (x + B > W ? W - x : B);
        const int blockHeight = (y + B > H ? H - y : B);

        const int xAct = (x > 0 ? 0 : bVal);
        const int yAct = (y > 0 ? 0 : bVal);
        const int wAct = (x + blockWidth < W ? blockWidth : blockWidth - bVal);
        const int hAct = (y + blockHeight < H ? blockHeight : blockHeight - bVal);

        if (wAct <= xAct || hAct <= yAct)
          continue;

        uint64_t res_c = sa8u_c(o, O, xAct, yAct, wAct, hAct);
        uint64_t res_sse = sa8u_sse(o, O, xAct, yAct, wAct, hAct);
        uint64_t res_avx2 = sa8u_avx2(o, O, xAct, yAct, wAct, hAct);

        EXPECT_EQ(res_c, res_sse);
        EXPECT_EQ(res_c, res_avx2);
      }
    }
  }
}

TEST_F(SaActTest, SaAct10u)
{
  auto sa10u_c = get_saact_func(xpsnr_cpu_c, 10, false);
  auto sa10u_sse = get_saact_func(xpsnr_cpu_sse41, 10, false);
  auto sa10u_avx2 = get_saact_func(xpsnr_cpu_avx2, 10, false);
  constexpr int bVal = 1;

  for (auto &tc : testset)
  {
    const int W = tc.width;
    const int H = tc.height;
    const int B = tc.get_block_size();

    for (int y = 0; y < H; y += B)
    {
      for (int x = 0; x < W; x += B)
      {
        const int O = tc.width * 2;
        const uint8_t* o = (uint8_t*)(tc.samples10u.data() + y * W + x);
        const int blockWidth = (x + B > W ? W - x : B);
        const int blockHeight = (y + B > H ? H - y : B);

        const int xAct = (x > 0 ? 0 : bVal);
        const int yAct = (y > 0 ? 0 : bVal);
        const int wAct = (x + blockWidth < W ? blockWidth : blockWidth - bVal);
        const int hAct = (y + blockHeight < H ? blockHeight : blockHeight - bVal);

        if (wAct <= xAct || hAct <= yAct)
          continue;

        uint64_t res_c = sa10u_c(o, O, xAct, yAct, wAct, hAct);
        uint64_t res_sse = sa10u_sse(o, O, xAct, yAct, wAct, hAct);
        uint64_t res_avx2 = sa10u_avx2(o, O, xAct, yAct, wAct, hAct);

        EXPECT_EQ(res_c, res_sse);
        EXPECT_EQ(res_c, res_avx2);
      }
    }
  }
}

TEST_F(SaActTest, SaAct8uDS)
{
  auto sa8uds_c = get_saact_func(xpsnr_cpu_c, 8, true);
  auto sa8uds_sse = get_saact_func(xpsnr_cpu_sse41, 8, true);
  auto sa8uds_avx2 = get_saact_func(xpsnr_cpu_avx2, 8, true);
  constexpr int bVal = 2;

  for (auto &tc : testset)
  {
    const int W = tc.width;
    const int H = tc.height;
    const int B = tc.get_block_size();

    for (int y = 0; y < H; y += B)
    {
      for (int x = 0; x < W; x += B)
      {
        const uint8_t* o = tc.samples8u.data() + y * W + x;
        const int blockWidth = (x + B > W ? W - x : B);
        const int blockHeight = (y + B > H ? H - y : B);

        const int xAct = (x > 0 ? 0 : bVal);
        const int yAct = (y > 0 ? 0 : bVal);
        const int wAct = (x + blockWidth < W ? blockWidth : blockWidth - bVal);
        const int hAct = (y + blockHeight < H ? blockHeight : blockHeight - bVal);

        if (wAct <= xAct || hAct <= yAct)
          continue;

        uint64_t res_c = sa8uds_c(o, W, xAct, yAct, wAct, hAct);
        uint64_t res_sse = sa8uds_sse(o, W, xAct, yAct, wAct, hAct);
        uint64_t res_avx2 = sa8uds_avx2(o, W, xAct, yAct, wAct, hAct);

        EXPECT_EQ(res_c, res_sse);
        EXPECT_EQ(res_c, res_avx2);
      }
    }
  }
}

TEST_F(SaActTest, SaAct10uDS)
{
  auto sa10uds_c = get_saact_func(xpsnr_cpu_c, 10, true);
  auto sa10uds_sse = get_saact_func(xpsnr_cpu_sse41, 10, true);
  auto sa10uds_avx2 = get_saact_func(xpsnr_cpu_avx2, 10, true);
  constexpr int bVal = 2;

  for (auto &tc : testset)
  {
    const int W = tc.width;
    const int H = tc.height;
    const int B = tc.get_block_size();

    for (int y = 0; y < H; y += B)
    {
      for (int x = 0; x < W; x += B)
      {
        const uint8_t* o = (uint8_t*)(tc.samples10u.data() + y * W + x);
        const int blockWidth = (x + B > W ? W - x : B);
        const int blockHeight = (y + B > H ? H - y : B);

        const int xAct = (x > 0 ? 0 : bVal);
        const int yAct = (y > 0 ? 0 : bVal);
        const int wAct = (x + blockWidth < W ? blockWidth : blockWidth - bVal);
        const int hAct = (y + blockHeight < H ? blockHeight : blockHeight - bVal);

        if (wAct <= xAct || hAct <= yAct)
          continue;

        uint64_t res_c = sa10uds_c(o, W * 2, xAct, yAct, wAct, hAct);
        uint64_t res_sse = sa10uds_sse(o, W * 2, xAct, yAct, wAct, hAct);
        uint64_t res_avx2 = sa10uds_avx2(o, W * 2, xAct, yAct, wAct, hAct);

        EXPECT_EQ(res_c, res_sse);
        EXPECT_EQ(res_c, res_avx2);
      }
    }
  }
}