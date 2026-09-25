// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef ENTERPRISE_STREAM_PIPELINE_HPP_
#define ENTERPRISE_STREAM_PIPELINE_HPP_

#include <vector>
#include <string>
#include "cuda_common.hpp"
#include "image_data.hpp"
#include "filter_kernels.hpp"

namespace enterprise_cuda {

struct StreamMetrics {
  int num_streams = 4;
  size_t total_items_processed = 0;
  double total_time_ms = 0.0;
  double average_item_time_ms = 0.0;
  double throughput_mpixels_per_sec = 0.0;
  double speedup_vs_synchronous = 1.0;
};

// Manages enterprise-scale multi-stream asynchronous batch processing.
class StreamPipeline {
 public:
  explicit StreamPipeline(int num_streams = 4);
  ~StreamPipeline();

  // Processes a batch of RGB images using asynchronous streams
  StreamMetrics ProcessBatchImages(const std::vector<ImageRGB>& input_batch,
                                   std::vector<ImageRGB>* output_batch,
                                   const std::string& filter_type,
                                   int kernel_size, float sigma, float angle);

  // Synchronous baseline for measuring pipelining speedup
  StreamMetrics ProcessBatchImagesSync(const std::vector<ImageRGB>& input_batch,
                                       std::vector<ImageRGB>* output_batch,
                                       const std::string& filter_type,
                                       int kernel_size, float sigma, float angle);

  int num_streams() const { return num_streams_; }

 private:
  int num_streams_;
#ifdef __CUDACC__
  std::vector<cudaStream_t> streams_;
#endif
};

}  // namespace enterprise_cuda

#endif  // ENTERPRISE_STREAM_PIPELINE_HPP_
