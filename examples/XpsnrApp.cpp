#include "xpsnr.hpp"
#include "yuv_reader.hpp"
#include "program_options_lite.h"

#include <string>
#include <vector>
#include <iostream>

struct app_configs
{
  std::string ref_path, dist_path;
  int width, height;
  double fps;
  
  bool parse(int argc, char** argv)
  {
    namespace po = df::program_options_lite;
    bool do_help = {};

    po::Options opts;
    opts.addOptions()
    ("help", do_help, false, "Print help text")
    ("-i", ref_path, std::string(), "reference input yuv path")
    ("-d", dist_path, std::string(), "distorded input yuv path")
    ("-w", width, 0, "yuv width")
    ("-h", height, 0, "yuv hegiht")
    ("-f", fps, 30.0, "frame rate");

    po::setDefaults(opts);
    po::ErrorReporter err;

    const auto &argv_unhandled = po::scanArgv(opts, argc, (const char **)argv, err);

    for (auto a : argv_unhandled)
    {
      std::cerr << "Unhandled argument ignored: " << a << std::endl;
    }

    if (argc == 1 || do_help)
    {
      po::doHelp(std::cout, opts);
      exit(0);
    }

    return true;
  }
};

int main(int argc, char** argv)
{
  app_configs config{};
  xpsnr xp;
  xpsnr_options_t xpsnr_opt{};

  std::unique_ptr<yuv_reader> yuv_ref, yuv_dist;

  std::vector<double> weights, wssds;
  std::vector<uint64_t> ssds;
  int blksize, nblk_w, nblk_h, poc = 0;

  config.parse(argc, argv);

  xpsnr_opt.width = config.width;
  xpsnr_opt.height = config.height;
  xpsnr_opt.bit_depth = 8;
  xpsnr_opt.frame_rate = config.fps;
  xpsnr_opt.cpu = xpsnr_cpu_sse41;
  
  xp.init(xpsnr_opt);
  
  yuv_ref.reset(new yuv_reader{config.ref_path, config.width, config.height, 8});
  yuv_dist.reset(new yuv_reader{config.dist_path, config.width, config.height, 8});

  
  const uint8_t* ref_ptr = nullptr;
  const uint8_t* dist_ptr = nullptr;

  int cnt = 0;
  do
  {
    ref_ptr = yuv_ref->read_next();
    dist_ptr = yuv_dist->read_next();
    
    xpsnr_putframe_exchanges_t ex {};
    ex.ref_ptr = ref_ptr;
    ex.dist_ptr = dist_ptr;
    
    xp.put_frame(ex);

    std::cout << cnt << ":" << ex.xpsnr << std::endl;
    ++cnt;

  } while (cnt < 100);

  return 0;
}