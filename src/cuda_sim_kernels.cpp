// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef __CUDACC__

#include "filter_kernels.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace enterprise_cuda {

namespace {

inline int SimClamp(int val, int max_dim) {
  return std::max(0, std::min(val, max_dim - 1));
}

inline void SimSwapIfGreater(uint8_t& a, uint8_t& b) {
  if (a > b) {
    uint8_t tmp = a;
    a = b;
    b = tmp;
  }
}

}  // namespace

// Global memory access simulation
float LaunchGaussianFilterNaive(const uint8_t* d_input, uint8_t* d_output,
                                int width, int height, int channels,
                                const float* h_kernel, int kernel_size,
                                cudaStream_t /*stream*/) {
  auto start = std::chrono::high_resolution_clock::now();
  int radius = kernel_size / 2;

  #pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -radius; ky <= radius; ++ky) {
          int sample_y = SimClamp(y + ky, height);
          for (int kx = -radius; kx <= radius; ++kx) {
            int sample_x = SimClamp(x + kx, width);
            float weight = h_kernel[(ky + radius) * kernel_size + (kx + radius)];
            int sample_idx = (sample_y * width + sample_x) * channels + c;
            sum += weight * static_cast<float>(d_input[sample_idx]);
          }
        }
        int out_idx = (y * width + x) * channels + c;
        const_cast<uint8_t*>(d_output)[out_idx] =
            static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, sum + 0.5f)));
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> elapsed = end - start;
  return elapsed.count();
}

// Tiled shared memory simulation
float LaunchGaussianFilterTiled(const uint8_t* d_input, uint8_t* d_output,
                                int width, int height, int channels,
                                const float* h_kernel, int kernel_size,
                                cudaStream_t stream) {
  // Executes identically to naive in terms of arithmetic results
  return LaunchGaussianFilterNaive(d_input, d_output, width, height, channels,
                                   h_kernel, kernel_size, stream);
}

// Constant memory simulation
float LaunchGaussianFilterConstant(const uint8_t* d_input, uint8_t* d_output,
                                   int width, int height, int channels,
                                   const float* h_kernel, int kernel_size,
                                   cudaStream_t stream) {
  return LaunchGaussianFilterNaive(d_input, d_output, width, height, channels,
                                   h_kernel, kernel_size, stream);
}

// Sobel filter simulation
float LaunchSobelFilter(const uint8_t* d_input, uint8_t* d_output,
                        int width, int height,
                        cudaStream_t /*stream*/) {
  auto start = std::chrono::high_resolution_clock::now();

  const int kGx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int kGy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  #pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int sum_x = 0;
      int sum_y = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        int sample_y = SimClamp(y + ky, height);
        for (int kx = -1; kx <= 1; ++kx) {
          int sample_x = SimClamp(x + kx, width);
          int val = d_input[sample_y * width + sample_x];
          sum_x += kGx[ky + 1][kx + 1] * val;
          sum_y += kGy[ky + 1][kx + 1] * val;
        }
      }
      float mag = std::sqrt(static_cast<float>(sum_x * sum_x + sum_y * sum_y));
      const_cast<uint8_t*>(d_output)[y * width + x] =
          static_cast<uint8_t>(std::min(255.0f, mag));
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> elapsed = end - start;
  return elapsed.count();
}

// Median filter simulation using the exact same 9-element sorting network
float LaunchMedianFilter(const uint8_t* d_input, uint8_t* d_output,
                         int width, int height, int channels,
                         cudaStream_t /*stream*/) {
  auto start = std::chrono::high_resolution_clock::now();

  #pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int xm1 = SimClamp(x - 1, width);
      int xp1 = SimClamp(x + 1, width);
      int ym1 = SimClamp(y - 1, height);
      int yp1 = SimClamp(y + 1, height);

      for (int c = 0; c < channels; ++c) {
        uint8_t v[9];
        v[0] = d_input[(ym1 * width + xm1) * channels + c];
        v[1] = d_input[(ym1 * width + x) * channels + c];
        v[2] = d_input[(ym1 * width + xp1) * channels + c];
        v[3] = d_input[(y * width + xm1) * channels + c];
        v[4] = d_input[(y * width + x) * channels + c];
        v[5] = d_input[(y * width + xp1) * channels + c];
        v[6] = d_input[(yp1 * width + xm1) * channels + c];
        v[7] = d_input[(yp1 * width + x) * channels + c];
        v[8] = d_input[(yp1 * width + xp1) * channels + c];

        SimSwapIfGreater(v[1], v[2]); SimSwapIfGreater(v[4], v[5]); SimSwapIfGreater(v[7], v[8]);
        SimSwapIfGreater(v[0], v[1]); SimSwapIfGreater(v[3], v[4]); SimSwapIfGreater(v[6], v[7]);
        SimSwapIfGreater(v[1], v[2]); SimSwapIfGreater(v[4], v[5]); SimSwapIfGreater(v[7], v[8]);
        SimSwapIfGreater(v[0], v[3]); SimSwapIfGreater(v[5], v[8]); SimSwapIfGreater(v[4], v[7]);
        SimSwapIfGreater(v[3], v[6]); SimSwapIfGreater(v[1], v[4]); SimSwapIfGreater(v[2], v[5]);
        SimSwapIfGreater(v[4], v[7]); SimSwapIfGreater(v[4], v[2]); SimSwapIfGreater(v[6], v[4]);
        SimSwapIfGreater(v[4], v[2]);

        const_cast<uint8_t*>(d_output)[(y * width + x) * channels + c] = v[4];
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> elapsed = end - start;
  return elapsed.count();
}

// Bilinear image rotation simulation
float LaunchImageRotation(const uint8_t* d_input, uint8_t* d_output,
                          int width, int height, int channels,
                          float angle_degrees,
                          cudaStream_t /*stream*/) {
  auto start = std::chrono::high_resolution_clock::now();

  float rad = -angle_degrees * static_cast<float>(M_PI / 180.0);
  float cos_theta = std::cos(rad);
  float sin_theta = std::sin(rad);
  float cx = width * 0.5f;
  float cy = height * 0.5f;

  #pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float dx = static_cast<float>(x) - cx;
      float dy = static_cast<float>(y) - cy;

      float src_x = cos_theta * dx - sin_theta * dy + cx;
      float src_y = sin_theta * dx + cos_theta * dy + cy;

      int out_idx_base = (y * width + x) * channels;

      if (src_x >= 0.0f && src_x < static_cast<float>(width - 1) &&
          src_y >= 0.0f && src_y < static_cast<float>(height - 1)) {
        int x0 = static_cast<int>(src_x);
        int y0 = static_cast<int>(src_y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float fx = src_x - static_cast<float>(x0);
        float fy = src_y - static_cast<float>(y0);
        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 = fx * (1.0f - fy);
        float w01 = (1.0f - fx) * fy;
        float w11 = fx * fy;

        for (int c = 0; c < channels; ++c) {
          float p00 = static_cast<float>(d_input[(y0 * width + x0) * channels + c]);
          float p10 = static_cast<float>(d_input[(y0 * width + x1) * channels + c]);
          float p01 = static_cast<float>(d_input[(y1 * width + x0) * channels + c]);
          float p11 = static_cast<float>(d_input[(y1 * width + x1) * channels + c]);
          float val = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
          const_cast<uint8_t*>(d_output)[out_idx_base + c] =
              static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, val + 0.5f)));
        }
      } else {
        for (int c = 0; c < channels; ++c) {
          const_cast<uint8_t*>(d_output)[out_idx_base + c] = 0;
        }
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> elapsed = end - start;
  return elapsed.count();
}

// 1D Signal convolution simulation
float LaunchBatchSignalConvolution1D(const float* d_signal, float* d_output,
                                    size_t signal_length,
                                    const float* d_kernel, size_t kernel_length,
                                    cudaStream_t /*stream*/) {
  auto start = std::chrono::high_resolution_clock::now();
  int radius = static_cast<int>(kernel_length / 2);

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < signal_length; ++i) {
    float sum = 0.0f;
    for (size_t k = 0; k < kernel_length; ++k) {
      int sample_idx = static_cast<int>(i) + static_cast<int>(k) - radius;
      if (sample_idx >= 0 && sample_idx < static_cast<int>(signal_length)) {
        sum += d_signal[sample_idx] * d_kernel[k];
      }
    }
    const_cast<float*>(d_output)[i] = sum;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> elapsed = end - start;
  return elapsed.count();
}

}  // namespace enterprise_cuda

#endif  // !__CUDACC__
