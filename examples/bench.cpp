#include "xpsnr.h"
#include <fstream>
#include <vector>

int run(int w, int h, double fps, const char* org_path, const char* dist_path, xpsnr_cpu_t cpu)
{
  const int s = w * h * 3 / 2;
  int block_size, nblk_inw, nblk_inh;
  std::ifstream org_file, dist_file;
  std::vector<uint8_t> org_buffer[3];
  std::vector<uint8_t> dist_buffer;
  std::vector<double> weights, wssd;
  std::vector<uint64_t> ssd;
  int count = 0;

  get_xpsnr_structures(w, h, fps, block_size, nblk_inw, nblk_inh);

  org_file.open(org_path, std::ios::binary);
  dist_file.open(dist_path, std::ios::binary);

  org_buffer[0].resize(w * h);
  org_buffer[1].resize(w * h);
  org_buffer[2].resize(w * h);
  dist_buffer.resize(w * h);
  weights.resize(nblk_inw * nblk_inh);
  wssd.resize(nblk_inw * nblk_inh);
  ssd.resize(nblk_inw * nblk_inh);
  
  while(true)
  {
    org_file.read((char*)org_buffer[count % 3].data(), w * h);
    dist_file.read((char*)dist_buffer.data(), w * h);

    if(org_file.gcount() != w * h || dist_file.gcount() != w * h)
    {
      break;
    }
    org_file.seekg(s - w * h , std::ios::cur);
    dist_file.seekg(s - w * h , std::ios::cur);

    calc_xpsnr_in_picture(w, h, 8, fps,
                          org_buffer[count % 3].data(),
                          dist_buffer.data(),
                          count > 0 ? org_buffer[(count - 1) % 3].data() : nullptr,
                          count > 1 ? org_buffer[(count - 2) % 3].data() : nullptr,
                          w, cpu, weights.data(), ssd.data(), wssd.data());

    ++count;
  }

  return 0;
}

int main(void)
{
  const char* org_path = "/storage/videos/yuv/jvet_1080p/FoodMarket4_1920x1080_60.yuv";
  const char* dist_path = "/storage/videos/yuv/jvet_1080p/FoodMarket4_dist.yuv";
  const int w = 1920;
  const int h = 1080; 
  const double fps = 60.0;
 
  run(w, h, fps, org_path, dist_path, xpsnr_cpu_avx2);

  return 0;
}