// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include "stream_pipeline.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>

#include "cpu_reference.hpp"

namespace enterprise_cuda {

StreamPipeline::StreamPipeline(int num_streams) : num_streams_(num_streams) {
#ifdef __CUDACC__
  streams_.resize(num_streams_);
  for (int i = 0; i < num_streams_; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams_[i]));
  }
#endif
}

StreamPipeline::~StreamPipeline() {
#ifdef __CUDACC__
  for (int i = 0; i < num_streams_; ++i) {
    cudaStreamDestroy(streams_[i]);
  }
#endif
}

StreamMetrics StreamPipeline::ProcessBatchImages(const std::vector<ImageRGB>& input_batch,
                                                 std::vector<ImageRGB>* output_batch,
                                                 const std::string& filter_type,
                                                 int kernel_size, float sigma, float angle) {
  StreamMetrics metrics;
  metrics.num_streams = num_streams_;
  metrics.total_items_processed = input_batch.size();
  output_batch->resize(input_batch.size());

  if (input_batch.empty()) return metrics;

  CpuTimer timer;
  timer.Reset();

  std::vector<float> kernel_weights = GenerateGaussianKernel2D(kernel_size, sigma);

#ifdef __CUDACC__
  // Ring of device buffers for each stream to allow simultaneous flight
  struct StreamSlot {
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    size_t allocated_bytes = 0;
  };

  std::vector<StreamSlot> slots(num_streams_);

  for (size_t i = 0; i < input_batch.size(); ++i) {
    int stream_idx = i % num_streams_;
    cudaStream_t stream = streams_[stream_idx];
    const ImageRGB& img = input_batch[i];
    size_t bytes = img.SizeInBytes();

    (*output_batch)[i].Allocate(img.width(), img.height());

    // Allocate or reallocate stream slots if needed
    if (slots[stream_idx].allocated_bytes < bytes) {
      if (slots[stream_idx].d_input) cudaFree(slots[stream_idx].d_input);
      if (slots[stream_idx].d_output) cudaFree(slots[stream_idx].d_output);
      CUDA_CHECK(cudaMalloc(&slots[stream_idx].d_input, bytes));
      CUDA_CHECK(cudaMalloc(&slots[stream_idx].d_output, bytes));
      slots[stream_idx].allocated_bytes = bytes;
    }

    // 1. Asynchronous Host-to-Device Transfer
    CUDA_CHECK(cudaMemcpyAsync(slots[stream_idx].d_input, img.data(), bytes,
                               cudaMemcpyHostToDevice, stream));

    // 2. Asynchronous Kernel Execution
    if (filter_type == "gaussian") {
      LaunchGaussianFilterTiled(slots[stream_idx].d_input, slots[stream_idx].d_output,
                                img.width(), img.height(), 3,
                                kernel_weights.data(), kernel_size, stream);
    } else if (filter_type == "median") {
      LaunchMedianFilter(slots[stream_idx].d_input, slots[stream_idx].d_output,
                         img.width(), img.height(), 3, stream);
    } else if (filter_type == "rotate") {
      LaunchImageRotation(slots[stream_idx].d_input, slots[stream_idx].d_output,
                          img.width(), img.height(), 3, angle, stream);
    } else {
      // Default to Tiled Gaussian
      LaunchGaussianFilterTiled(slots[stream_idx].d_input, slots[stream_idx].d_output,
                                img.width(), img.height(), 3,
                                kernel_weights.data(), kernel_size, stream);
    }

    // 3. Asynchronous Device-to-Host Transfer
    CUDA_CHECK(cudaMemcpyAsync((*output_batch)[i].data(), slots[stream_idx].d_output,
                               bytes, cudaMemcpyDeviceToHost, stream));
  }

  // Synchronize all active streams
  for (int s = 0; s < num_streams_; ++s) {
    CUDA_CHECK(cudaStreamSynchronize(streams_[s]));
    if (slots[s].d_input) cudaFree(slots[s].d_input);
    if (slots[s].d_output) cudaFree(slots[s].d_output);
  }

#else
  // Simulation: Process items across threads
  for (size_t i = 0; i < input_batch.size(); ++i) {
    const ImageRGB& img = input_batch[i];
    (*output_batch)[i].Allocate(img.width(), img.height());

    if (filter_type == "gaussian") {
      LaunchGaussianFilterTiled(img.data(), (*output_batch)[i].data(),
                                img.width(), img.height(), 3,
                                kernel_weights.data(), kernel_size, nullptr);
    } else if (filter_type == "median") {
      LaunchMedianFilter(img.data(), (*output_batch)[i].data(),
                         img.width(), img.height(), 3, nullptr);
    } else if (filter_type == "rotate") {
      LaunchImageRotation(img.data(), (*output_batch)[i].data(),
                          img.width(), img.height(), 3, angle, nullptr);
    } else {
      LaunchGaussianFilterTiled(img.data(), (*output_batch)[i].data(),
                                img.width(), img.height(), 3,
                                kernel_weights.data(), kernel_size, nullptr);
    }
  }
#endif

  metrics.total_time_ms = timer.ElapsedMilliseconds();
  metrics.average_item_time_ms = metrics.total_time_ms / input_batch.size();

  size_t total_pixels = 0;
  for (const auto& img : input_batch) {
    total_pixels += img.TotalPixels();
  }
  double total_seconds = metrics.total_time_ms / 1000.0;
  if (total_seconds > 0.0) {
    metrics.throughput_mpixels_per_sec = (total_pixels / 1e6) / total_seconds;
  }

  return metrics;
}

StreamMetrics StreamPipeline::ProcessBatchImagesSync(const std::vector<ImageRGB>& input_batch,
                                                     std::vector<ImageRGB>* output_batch,
                                                     const std::string& filter_type,
                                                     int kernel_size, float sigma, float angle) {
  StreamMetrics metrics;
  metrics.num_streams = 1;
  metrics.total_items_processed = input_batch.size();
  output_batch->resize(input_batch.size());

  if (input_batch.empty()) return metrics;

  CpuTimer timer;
  timer.Reset();

  std::vector<float> kernel_weights = GenerateGaussianKernel2D(kernel_size, sigma);

#ifdef __CUDACC__
  uint8_t* d_input = nullptr;
  uint8_t* d_output = nullptr;
  size_t max_bytes = 0;
  for (const auto& img : input_batch) {
    if (img.SizeInBytes() > max_bytes) max_bytes = img.SizeInBytes();
  }

  CUDA_CHECK(cudaMalloc(&d_input, max_bytes));
  CUDA_CHECK(cudaMalloc(&d_output, max_bytes));

  for (size_t i = 0; i < input_batch.size(); ++i) {
    const ImageRGB& img = input_batch[i];
    size_t bytes = img.SizeInBytes();
    (*output_batch)[i].Allocate(img.width(), img.height());

    // Synchronous blocking HtoD
    CUDA_CHECK(cudaMemcpy(d_input, img.data(), bytes, cudaMemcpyHostToDevice));

    // Kernel execution on default stream
    if (filter_type == "gaussian") {
      LaunchGaussianFilterTiled(d_input, d_output, img.width(), img.height(), 3,
                                kernel_weights.data(), kernel_size, 0);
    } else if (filter_type == "median") {
      LaunchMedianFilter(d_input, d_output, img.width(), img.height(), 3, 0);
    } else if (filter_type == "rotate") {
      LaunchImageRotation(d_input, d_output, img.width(), img.height(), 3, angle, 0);
    } else {
      LaunchGaussianFilterTiled(d_input, d_output, img.width(), img.height(), 3,
                                kernel_weights.data(), kernel_size, 0);
    }

    // Synchronous blocking DtoH
    CUDA_CHECK(cudaMemcpy((*output_batch)[i].data(), d_output, bytes, cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
#else
  for (size_t i = 0; i < input_batch.size(); ++i) {
    const ImageRGB& img = input_batch[i];
    (*output_batch)[i].Allocate(img.width(), img.height());
    if (filter_type == "gaussian") {
      LaunchGaussianFilterNaive(img.data(), (*output_batch)[i].data(),
                                img.width(), img.height(), 3,
                                kernel_weights.data(), kernel_size, nullptr);
    } else if (filter_type == "median") {
      LaunchMedianFilter(img.data(), (*output_batch)[i].data(),
                         img.width(), img.height(), 3, nullptr);
    } else if (filter_type == "rotate") {
      LaunchImageRotation(img.data(), (*output_batch)[i].data(),
                          img.width(), img.height(), 3, angle, nullptr);
    } else {
      LaunchGaussianFilterNaive(img.data(), (*output_batch)[i].data(),
                                img.width(), img.height(), 3,
                                kernel_weights.data(), kernel_size, nullptr);
    }
  }
#endif

  metrics.total_time_ms = timer.ElapsedMilliseconds();
  metrics.average_item_time_ms = metrics.total_time_ms / input_batch.size();

  size_t total_pixels = 0;
  for (const auto& img : input_batch) {
    total_pixels += img.TotalPixels();
  }
  double total_seconds = metrics.total_time_ms / 1000.0;
  if (total_seconds > 0.0) {
    metrics.throughput_mpixels_per_sec = (total_pixels / 1e6) / total_seconds;
  }

  return metrics;
}

}  // namespace enterprise_cuda
