#include "distortion.h"

#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>

struct dist_test_case_t
{
  int width;
  int height;
  std::vector<uint8_t> samples8u[2];
  std::vector<uint16_t> samples10u[2];

  void resize(size_t size)
  {
    samples8u[0].resize(size);
    samples8u[1].resize(size);
    samples10u[0].resize(size);
    samples10u[1].resize(size);
  }
};

class DistotionTest: public testing::Test
{
protected:
  std::vector<dist_test_case_t> testset;
  
  void SetUp() override
  {
    std::mt19937 gen{1313};
    std::uniform_int_distribution<uint8_t> dis8u(0, 255);
    std::uniform_int_distribution<uint16_t> dis10u(0, 1023);

    testset.clear();

    constexpr int resolutions[5][2]{
        {128, 128},
        {64, 64},
        {32, 32},
        {16, 16},
        {8, 8}
    };

    for (int ridx = 0; ridx < 5; ++ridx)
    {
      int w = resolutions[ridx][0];
      int h = resolutions[ridx][1];

      for (int i = 0; i < 256; ++i)
      {
        dist_test_case_t tc{};
        tc.width = w;
        tc.height = h;
        tc.resize(w * h);

        for (int j = 0; j < w * h; ++j)
        {
          tc.samples8u[0][j] = dis8u(gen);
          tc.samples8u[1][j] = dis8u(gen);
          tc.samples10u[0][j] = dis10u(gen);
          tc.samples10u[1][j] = dis10u(gen);
        }

        testset.emplace_back(std::forward<decltype(tc)>(tc));
      }
    }
  }
};

TEST_F(DistotionTest, DistotionSSD8u)
{
  for (auto &tc : testset)
  {
    const int W = tc.width;
    const int H = tc.height;

    uint64_t res_c = ssd_c(tc.samples8u[0].data(), tc.samples8u[1].data(), W, W, H);
    uint64_t res_sse = ssd8u_sse(tc.samples8u[0].data(), tc.samples8u[1].data(), W, W, H);
    uint64_t res_avx2 = ssd8u_avx2(tc.samples8u[0].data(), tc.samples8u[1].data(), W, W, H);

    EXPECT_EQ(res_c, res_sse);
    EXPECT_EQ(res_c, res_avx2);
  }
}

TEST_F(DistotionTest, DistotionSSD10u)
{
  for (auto &tc : testset)
  {
    const int W = tc.width;
    const int H = tc.height;

    uint64_t res_c = ssd_c(tc.samples10u[0].data(), tc.samples10u[1].data(), W * 2, W, H);
    uint64_t res_sse = ssd10u_sse((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), W * 2, W, H);
    uint64_t res_avx2 = ssd10u_avx2((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), W * 2, W, H);

    EXPECT_EQ(res_c, res_sse);
    EXPECT_EQ(res_c, res_avx2);
  }
}
