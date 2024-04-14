#include "ta_act.h"

#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>

struct test_case_t
{
  std::vector<uint8_t> samples8u[3];
  std::vector<uint16_t> samples10u[3];
  int width;
  int height;

  void resize(size_t size)
  {
    samples8u[0].resize(size);
    samples8u[1].resize(size);
    samples8u[2].resize(size);
    samples10u[0].resize(size);
    samples10u[1].resize(size);
    samples10u[2].resize(size);
  }
};

class TaActTest: public testing::Test
{
protected:
  std::vector<test_case_t> testset;
  
  void SetUp() override
  {
    std::mt19937 gen{1111};
    std::uniform_int_distribution<int> dis8u(0, 255);
    std::uniform_int_distribution<int> dis10u(0, 1023);

    constexpr int resolutions[6][2]{
        {4, 4},
        {8, 8},
        {16, 16},
        {32, 32},
        {64, 64},
        {128, 128},
    };

    for (int ridx = 0; ridx < 6; ++ridx)
    {
      int w = resolutions[ridx][0];
      int h = resolutions[ridx][1];

      for (int i = 0; i < 256; ++i)
      {
        test_case_t tc;
        tc.resize(w * h);

        for (int j = 0; j < w * h; ++j)
        {
          tc.samples8u[0][j] = (uint8_t)dis8u(gen);
          tc.samples8u[1][j] = (uint8_t)dis8u(gen);
          tc.samples8u[2][j] = (uint8_t)dis8u(gen);
          tc.samples10u[0][j] = (uint16_t)dis10u(gen);
          tc.samples10u[1][j] = (uint16_t)dis10u(gen);
          tc.samples10u[2][j] = (uint16_t)dis10u(gen);
        }
        tc.width = w;
        tc.height = h;

        testset.emplace_back(tc);
      }
    }
  }
};

TEST_F(TaActTest, TaAct8uO1) 
{
  auto tao1_8u_c = get_temp_act_func(8, 1, false);
  auto tao1_8u_sse = get_temp_act_sse_func(8, 1, false);
  auto tao1_8u_avx2 = get_temp_act_avx2_func(8, 1, false);

  for (auto &tc : testset)
  {
    uint64_t res_c = tao1_8u_c(tc.samples8u[0].data(), tc.samples8u[1].data(), nullptr, tc.width, tc.width, tc.height);
    uint64_t res_sse = tao1_8u_sse(tc.samples8u[0].data(), tc.samples8u[1].data(), nullptr, tc.width, tc.width, tc.height);
    uint64_t res_avx2 = tao1_8u_avx2(tc.samples8u[0].data(), tc.samples8u[1].data(), nullptr, tc.width, tc.width, tc.height);

    EXPECT_EQ(res_c, res_sse);
    EXPECT_EQ(res_c, res_avx2);
  }
}

TEST_F(TaActTest, TaAct10uO1) 
{
  auto tao1_10u_c = get_temp_act_func(10, 1, false);
  auto tao1_10u_sse = get_temp_act_sse_func(10, 1, false);
  auto tao1_10u_avx2 = get_temp_act_avx2_func(10, 1, false);

  for (auto &tc : testset)
  {
    uint64_t res_c = tao1_10u_c((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), nullptr, tc.width * 2, tc.width, tc.height);
    uint64_t res_sse = tao1_10u_sse((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), nullptr, tc.width * 2, tc.width, tc.height);
    uint64_t res_avx2 = tao1_10u_avx2((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), nullptr, tc.width * 2, tc.width, tc.height);

    EXPECT_EQ(res_c, res_sse);
    EXPECT_EQ(res_c, res_avx2);
  }
}

TEST_F(TaActTest, TaAct8uO2)
{
  auto tao2_8u_c = get_temp_act_func(8, 2, false);
  auto tao2_8u_sse = get_temp_act_sse_func(8, 2, false);
  auto tao2_8u_avx2 = get_temp_act_avx2_func(8, 2, false);

  for (auto &tc : testset)
  {
    uint64_t res_c = tao2_8u_c(tc.samples8u[0].data(), tc.samples8u[1].data(), tc.samples8u[2].data(), tc.width, tc.width, tc.height);
    uint64_t res_sse = tao2_8u_sse(tc.samples8u[0].data(), tc.samples8u[1].data(), tc.samples8u[2].data(), tc.width, tc.width, tc.height);
    uint64_t res_avx2 = tao2_8u_avx2(tc.samples8u[0].data(), tc.samples8u[1].data(), tc.samples8u[2].data(), tc.width, tc.width, tc.height);

    EXPECT_EQ(res_c, res_sse);
    EXPECT_EQ(res_c, res_avx2);
  }
}

TEST_F(TaActTest, TaAct10uO2) 
{
  auto tao2_10u_c = get_temp_act_func(10, 2, false);
  auto tao2_10u_sse = get_temp_act_sse_func(10, 2, false);
  auto tao2_10u_avx2 = get_temp_act_avx2_func(10, 2, false);

  for (auto &tc : testset)
  {
    uint64_t res_c = tao2_10u_c((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), (uint8_t *)tc.samples10u[2].data(), tc.width * 2, tc.width, tc.height);
    uint64_t res_sse = tao2_10u_sse((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), (uint8_t *)tc.samples10u[2].data(), tc.width * 2, tc.width, tc.height);
    uint64_t res_avx2 = tao2_10u_avx2((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), (uint8_t *)tc.samples10u[2].data(), tc.width * 2, tc.width, tc.height);

    EXPECT_EQ(res_c, res_sse);
    EXPECT_EQ(res_c, res_avx2);
  }
}

TEST_F(TaActTest, TaAct8uO1Ds)
{
  auto tao1_8uds_c = get_temp_act_func(8, 1, true);
  auto tao1_8uds_sse = get_temp_act_sse_func(8, 1, true);
  auto tao1_8uds_avx2 = get_temp_act_avx2_func(8, 1, true);

  for (auto &tc : testset)
  {
    uint64_t res_c = tao1_8uds_c(tc.samples8u[0].data(), tc.samples8u[1].data(), nullptr, tc.width, tc.width, tc.height);
    uint64_t res_sse = tao1_8uds_sse(tc.samples8u[0].data(), tc.samples8u[1].data(), nullptr, tc.width, tc.width, tc.height);
    uint64_t res_avx2 = tao1_8uds_avx2(tc.samples8u[0].data(), tc.samples8u[1].data(), nullptr, tc.width, tc.width, tc.height);

    EXPECT_EQ(res_c, res_sse);
    EXPECT_EQ(res_c, res_avx2);
  }
}

TEST_F(TaActTest, TaAct10uO1Ds)
{
  auto tao1_10uds_c = get_temp_act_func(10, 1, true);
  auto tao1_10uds_sse = get_temp_act_sse_func(10, 1, true);
  auto tao1_10uds_avx2 = get_temp_act_avx2_func(10, 1, true);

  for (auto &tc : testset)
  {
    uint64_t res_c = tao1_10uds_c((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), nullptr, tc.width * 2, tc.width, tc.height);
    uint64_t res_sse = tao1_10uds_sse((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), nullptr, tc.width * 2, tc.width, tc.height);
    uint64_t res_avx2 = tao1_10uds_avx2((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), nullptr, tc.width * 2, tc.width, tc.height);

    EXPECT_EQ(res_c, res_sse);
  }
}

TEST_F(TaActTest, TaAct8uO2Ds)
{
  auto tao2_8uds_c = get_temp_act_func(8, 2, true);
  auto tao2_8uds_sse = get_temp_act_sse_func(8, 2, true);
  auto tao2_8uds_avx2 = get_temp_act_avx2_func(8, 2, true);

  for (auto &tc : testset)
  {
    uint64_t res_c = tao2_8uds_c(tc.samples8u[0].data(), tc.samples8u[1].data(), tc.samples8u[2].data(), tc.width, tc.width, tc.height);
    uint64_t res_sse = tao2_8uds_sse(tc.samples8u[0].data(), tc.samples8u[1].data(), tc.samples8u[2].data(), tc.width, tc.width, tc.height);
    uint64_t res_avx2 = tao2_8uds_avx2(tc.samples8u[0].data(), tc.samples8u[1].data(), tc.samples8u[2].data(), tc.width, tc.width, tc.height);

    EXPECT_EQ(res_c, res_sse);
    EXPECT_EQ(res_c, res_avx2);
  }
}

TEST_F(TaActTest, TaAct10uO2Ds)
{
  auto tao2_10uds_c = get_temp_act_func(10, 2, true);
  auto tao2_10uds_sse = get_temp_act_sse_func(10, 2, true);
  auto tao2_10uds_avx2 = get_temp_act_avx2_func(10, 2, true);

  for (auto &tc : testset)
  {
    uint64_t res_c = tao2_10uds_c((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), (uint8_t *)tc.samples10u[2].data(), tc.width * 2, tc.width, tc.height);
    uint64_t res_sse = tao2_10uds_sse((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), (uint8_t *)tc.samples10u[2].data(), tc.width * 2, tc.width, tc.height);
    uint64_t res_avx2 = tao2_10uds_avx2((uint8_t *)tc.samples10u[0].data(), (uint8_t *)tc.samples10u[1].data(), (uint8_t *)tc.samples10u[2].data(), tc.width * 2, tc.width, tc.height);

    EXPECT_EQ(res_c, res_sse);
    EXPECT_EQ(res_c, res_avx2);
  }
}