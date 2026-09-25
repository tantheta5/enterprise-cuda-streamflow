// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef ENTERPRISE_CLI_PARSER_HPP_
#define ENTERPRISE_CLI_PARSER_HPP_

#include <getopt.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace enterprise_cuda {

struct CommandLineOptions {
  std::string mode = "all";                // "image", "signal", "all", "benchmark", "verify"
  std::string input_path = "data/input";   // file or directory
  std::string output_path = "data/output"; // destination directory
  std::string filter_type = "all";         // "gaussian", "sobel", "median", "rotate", "convolve", "all"
  int kernel_size = 5;                     // 3, 5, 7, 9 (must be odd)
  float gaussian_sigma = 1.6f;             // standard deviation
  float rotation_angle = 45.0f;            // degrees
  int num_streams = 4;                     // concurrent CUDA streams
  int device_id = 0;                       // CUDA device ID
  int iterations = 10;                     // iterations for timing benchmarks
  bool run_benchmark = false;              // trigger benchmark matrix
  bool run_verify = false;                 // compare against CPU reference
  bool verbose = true;                     // print detailed step timings
};

inline void PrintUsage(const char* program_name) {
  std::printf("\n================================================================================\n");
  std::printf(" Enterprise CUDA StreamFlow: Asynchronous Batch Image & Signal Pipeline\n");
  std::printf(" Course Project: CUDA at Scale for the Enterprise\n");
  std::printf("================================================================================\n");
  std::printf("Usage: %s [options]\n\n", program_name);
  std::printf("Options:\n");
  std::printf("  -m, --mode <mode>         Execution mode: 'image', 'signal', 'all', 'benchmark', 'verify' (default: all)\n");
  std::printf("  -i, --input <path>        Input file or directory (default: 'data/input')\n");
  std::printf("  -o, --output <path>       Output directory for processed artifacts (default: 'data/output')\n");
  std::printf("  -f, --filter <type>       Filter type: 'gaussian', 'sobel', 'median', 'rotate', 'convolve', 'all' (default: all)\n");
  std::printf("  -k, --kernel-size <int>   Kernel window size for 2D filter (3, 5, 7, 9) (default: 5)\n");
  std::printf("  -s, --sigma <float>       Gaussian standard deviation sigma (default: 1.6)\n");
  std::printf("  -a, --angle <float>       Image rotation angle in degrees (default: 45.0)\n");
  std::printf("  -n, --streams <int>       Number of concurrent CUDA streams for pipelining (1 to 16) (default: 4)\n");
  std::printf("  -d, --device <int>        CUDA GPU device index (default: 0)\n");
  std::printf("  -c, --iterations <int>    Averaging iterations for benchmarking (default: 10)\n");
  std::printf("  -b, --benchmark           Run enterprise scalability benchmark across resolutions & streams\n");
  std::printf("  -v, --verify              Verify bit-accurate numerical parity between CPU and GPU\n");
  std::printf("  -q, --quiet               Suppress verbose per-item timing output\n");
  std::printf("  -h, --help                Display this detailed help message and exit\n");
  std::printf("\nExamples:\n");
  std::printf("  %s --mode all --streams 4\n", program_name);
  std::printf("  %s --input data/input/images/lena_512.ppm --filter gaussian --kernel-size 5\n", program_name);
  std::printf("  %s --benchmark --iterations 20\n", program_name);
  std::printf("  %s --verify\n\n", program_name);
}

inline CommandLineOptions ParseCommandLineArgs(int argc, char* argv[]) {
  CommandLineOptions options;

  static struct option long_options[] = {
      {"mode", required_argument, nullptr, 'm'},
      {"input", required_argument, nullptr, 'i'},
      {"output", required_argument, nullptr, 'o'},
      {"filter", required_argument, nullptr, 'f'},
      {"kernel-size", required_argument, nullptr, 'k'},
      {"sigma", required_argument, nullptr, 's'},
      {"angle", required_argument, nullptr, 'a'},
      {"streams", required_argument, nullptr, 'n'},
      {"device", required_argument, nullptr, 'd'},
      {"iterations", required_argument, nullptr, 'c'},
      {"benchmark", no_argument, nullptr, 'b'},
      {"verify", no_argument, nullptr, 'v'},
      {"quiet", no_argument, nullptr, 'q'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "m:i:o:f:k:s:a:n:d:c:bvqh", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'm':
        options.mode = optarg;
        break;
      case 'i':
        options.input_path = optarg;
        break;
      case 'o':
        options.output_path = optarg;
        break;
      case 'f':
        options.filter_type = optarg;
        break;
      case 'k':
        options.kernel_size = std::atoi(optarg);
        if (options.kernel_size < 3 || options.kernel_size % 2 == 0) {
          std::fprintf(stderr, "Warning: kernel-size must be an odd integer >= 3. Setting to 5.\n");
          options.kernel_size = 5;
        }
        break;
      case 's':
        options.gaussian_sigma = static_cast<float>(std::atof(optarg));
        break;
      case 'a':
        options.rotation_angle = static_cast<float>(std::atof(optarg));
        break;
      case 'n':
        options.num_streams = std::atoi(optarg);
        if (options.num_streams < 1) options.num_streams = 1;
        break;
      case 'd':
        options.device_id = std::atoi(optarg);
        break;
      case 'c':
        options.iterations = std::atoi(optarg);
        if (options.iterations < 1) options.iterations = 1;
        break;
      case 'b':
        options.run_benchmark = true;
        break;
      case 'v':
        options.run_verify = true;
        break;
      case 'q':
        options.verbose = false;
        break;
      case 'h':
        PrintUsage(argv[0]);
        std::exit(EXIT_SUCCESS);
      default:
        PrintUsage(argv[0]);
        std::exit(EXIT_FAILURE);
    }
  }

  return options;
}

}  // namespace enterprise_cuda

#endif  // ENTERPRISE_CLI_PARSER_HPP_
