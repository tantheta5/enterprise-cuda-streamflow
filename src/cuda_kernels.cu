// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include "filter_kernels.hpp"

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>

namespace enterprise_cuda {

// Fast device clamp helper
__device__ inline int DevClamp(int val, int max_dim) {
  return max(0, min(val, max_dim - 1));
}

// Constant memory cache for 2D convolution kernel mask (broadcast access)
__constant__ float c_filter_kernel[kMaxConstantKernelElements];

// =============================================================================
// 1. Naive 2D Gaussian Blur Kernel (Global Memory Access)
// =============================================================================
__global__ void GaussianFilterNaiveKernel(const uint8_t* __restrict__ input,
                                          uint8_t* __restrict__ output,
                                          int width, int height, int channels,
                                          const float* __restrict__ kernel,
                                          int kernel_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int radius = kernel_size / 2;

  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky) {
      int sample_y = DevClamp(y + ky, height);
      for (int kx = -radius; kx <= radius; ++kx) {
        int sample_x = DevClamp(x + kx, width);
        float weight = kernel[(ky + radius) * kernel_size + (kx + radius)];
        int sample_idx = (sample_y * width + sample_x) * channels + c;
        sum += weight * static_cast<float>(input[sample_idx]);
      }
    }
    int out_idx = (y * width + x) * channels + c;
    output[out_idx] = static_cast<uint8_t>(min(255.0f, max(0.0f, sum + 0.5f)));
  }
}

// =============================================================================
// 2. Optimized Tiled 2D Gaussian Blur Kernel (Shared Memory with Apron)
// =============================================================================
constexpr int kSharedTileDim = 16;
constexpr int kSharedApronRadius = 2; // radius 2 for 5x5 kernel
constexpr int kSharedBlockDim = kSharedTileDim + 2 * kSharedApronRadius; // 20x20

__global__ void GaussianFilterTiledKernel(const uint8_t* __restrict__ input,
                                          uint8_t* __restrict__ output,
                                          int width, int height, int channels,
                                          const float* __restrict__ kernel,
                                          int kernel_size) {
  // Shared memory tile containing pixel data + halo apron cells
  __shared__ float s_data[kSharedBlockDim][kSharedBlockDim];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int out_x = blockIdx.x * kSharedTileDim + tx;
  int out_y = blockIdx.y * kSharedTileDim + ty;

  int radius = kernel_size / 2;

  for (int c = 0; c < channels; ++c) {
    // Collaborative loading of shared memory tile including halo apron
    // Each thread in the 16x16 block participates in loading the 20x20 tile
    int linear_tid = ty * kSharedTileDim + tx;
    int total_shared_elements = kSharedBlockDim * kSharedBlockDim; // 400

    for (int idx = linear_tid; idx < total_shared_elements; idx += (kSharedTileDim * kSharedTileDim)) {
      int s_y = idx / kSharedBlockDim;
      int s_x = idx % kSharedBlockDim;

      int global_x = DevClamp(blockIdx.x * kSharedTileDim + s_x - radius, width);
      int global_y = DevClamp(blockIdx.y * kSharedTileDim + s_y - radius, height);

      int in_idx = (global_y * width + global_x) * channels + c;
      s_data[s_y][s_x] = static_cast<float>(input[in_idx]);
    }

    __syncthreads();

    if (out_x < width && out_y < height) {
      float sum = 0.0f;
      #pragma unroll
      for (int ky = -radius; ky <= radius; ++ky) {
        #pragma unroll
        for (int kx = -radius; kx <= radius; ++kx) {
          float weight = kernel[(ky + radius) * kernel_size + (kx + radius)];
          sum += weight * s_data[ty + radius + ky][tx + radius + kx];
        }
      }
      int out_idx = (out_y * width + out_x) * channels + c;
      output[out_idx] = static_cast<uint8_t>(min(255.0f, max(0.0f, sum + 0.5f)));
    }

    __syncthreads();
  }
}

// =============================================================================
// 3. Constant-Memory 2D Gaussian Kernel (Constant Cache Broadcast)
// =============================================================================
__global__ void GaussianFilterConstantKernel(const uint8_t* __restrict__ input,
                                             uint8_t* __restrict__ output,
                                             int width, int height, int channels,
                                             int kernel_size) {
  __shared__ float s_data[kSharedBlockDim][kSharedBlockDim];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int out_x = blockIdx.x * kSharedTileDim + tx;
  int out_y = blockIdx.y * kSharedTileDim + ty;

  int radius = kernel_size / 2;

  for (int c = 0; c < channels; ++c) {
    int linear_tid = ty * kSharedTileDim + tx;
    int total_shared_elements = kSharedBlockDim * kSharedBlockDim;

    for (int idx = linear_tid; idx < total_shared_elements; idx += (kSharedTileDim * kSharedTileDim)) {
      int s_y = idx / kSharedBlockDim;
      int s_x = idx % kSharedBlockDim;

      int global_x = DevClamp(blockIdx.x * kSharedTileDim + s_x - radius, width);
      int global_y = DevClamp(blockIdx.y * kSharedTileDim + s_y - radius, height);

      int in_idx = (global_y * width + global_x) * channels + c;
      s_data[s_y][s_x] = static_cast<float>(input[in_idx]);
    }

    __syncthreads();

    if (out_x < width && out_y < height) {
      float sum = 0.0f;
      #pragma unroll
      for (int ky = -radius; ky <= radius; ++ky) {
        #pragma unroll
        for (int kx = -radius; kx <= radius; ++kx) {
          // Access filter weights through constant cache broadcast
          float weight = c_filter_kernel[(ky + radius) * kernel_size + (kx + radius)];
          sum += weight * s_data[ty + radius + ky][tx + radius + kx];
        }
      }
      int out_idx = (out_y * width + out_x) * channels + c;
      output[out_idx] = static_cast<uint8_t>(min(255.0f, max(0.0f, sum + 0.5f)));
    }

    __syncthreads();
  }
}

// =============================================================================
// 4. Sobel 3x3 Filter Kernel with Fused Gradient Magnitude
// =============================================================================
__global__ void SobelFilterKernel(const uint8_t* __restrict__ input,
                                  uint8_t* __restrict__ output,
                                  int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  // Neighbor sample coordinates with border clamp
  int xm1 = DevClamp(x - 1, width);
  int xp1 = DevClamp(x + 1, width);
  int ym1 = DevClamp(y - 1, height);
  int yp1 = DevClamp(y + 1, height);

  // Read 3x3 neighborhood
  int p00 = input[ym1 * width + xm1];
  int p01 = input[ym1 * width + x];
  int p02 = input[ym1 * width + xp1];

  int p10 = input[y * width + xm1];
  int p12 = input[y * width + xp1];

  int p20 = input[yp1 * width + xm1];
  int p21 = input[yp1 * width + x];
  int p22 = input[yp1 * width + xp1];

  // Sobel Gx and Gy
  int gx = (-p00 + p02) + 2 * (-p10 + p12) + (-p20 + p22);
  int gy = (-p00 - 2 * p01 - p02) + (p20 + 2 * p21 + p22);

  float mag = sqrtf(static_cast<float>(gx * gx + gy * gy));
  output[y * width + x] = static_cast<uint8_t>(min(255.0f, mag));
}

// =============================================================================
// 5. Median 3x3 Non-Linear Filter Kernel (Sorting Network)
// =============================================================================
__device__ inline void SwapIfGreater(uint8_t& a, uint8_t& b) {
  if (a > b) {
    uint8_t tmp = a;
    a = b;
    b = tmp;
  }
}

__global__ void MedianFilterKernel(const uint8_t* __restrict__ input,
                                   uint8_t* __restrict__ output,
                                   int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int xm1 = DevClamp(x - 1, width);
  int xp1 = DevClamp(x + 1, width);
  int ym1 = DevClamp(y - 1, height);
  int yp1 = DevClamp(y + 1, height);

  for (int c = 0; c < channels; ++c) {
    uint8_t v[9];
    v[0] = input[(ym1 * width + xm1) * channels + c];
    v[1] = input[(ym1 * width + x) * channels + c];
    v[2] = input[(ym1 * width + xp1) * channels + c];
    v[3] = input[(y * width + xm1) * channels + c];
    v[4] = input[(y * width + x) * channels + c];
    v[5] = input[(y * width + xp1) * channels + c];
    v[6] = input[(yp1 * width + xm1) * channels + c];
    v[7] = input[(yp1 * width + x) * channels + c];
    v[8] = input[(yp1 * width + xp1) * channels + c];

    // Optimal 9-element partial sorting network to extract median v[4]
    SwapIfGreater(v[1], v[2]); SwapIfGreater(v[4], v[5]); SwapIfGreater(v[7], v[8]);
    SwapIfGreater(v[0], v[1]); SwapIfGreater(v[3], v[4]); SwapIfGreater(v[6], v[7]);
    SwapIfGreater(v[1], v[2]); SwapIfGreater(v[4], v[5]); SwapIfGreater(v[7], v[8]);
    SwapIfGreater(v[0], v[3]); SwapIfGreater(v[5], v[8]); SwapIfGreater(v[4], v[7]);
    SwapIfGreater(v[3], v[6]); SwapIfGreater(v[1], v[4]); SwapIfGreater(v[2], v[5]);
    SwapIfGreater(v[4], v[7]); SwapIfGreater(v[4], v[2]); SwapIfGreater(v[6], v[4]);
    SwapIfGreater(v[4], v[2]);

    output[(y * width + x) * channels + c] = v[4];
  }
}

// =============================================================================
// 6. Bilinear Image Rotation Kernel (Inverse Coordinate Mapping)
// =============================================================================
__global__ void ImageRotationKernel(const uint8_t* __restrict__ input,
                                    uint8_t* __restrict__ output,
                                    int width, int height, int channels,
                                    float cos_theta, float sin_theta,
                                    float cx, float cy) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  float dx = static_cast<float>(x) - cx;
  float dy = static_cast<float>(y) - cy;

  // Inverse coordinate mapping
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
      float p00 = static_cast<float>(input[(y0 * width + x0) * channels + c]);
      float p10 = static_cast<float>(input[(y0 * width + x1) * channels + c]);
      float p01 = static_cast<float>(input[(y1 * width + x0) * channels + c]);
      float p11 = static_cast<float>(input[(y1 * width + x1) * channels + c]);
      float val = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
      output[out_idx_base + c] = static_cast<uint8_t>(min(255.0f, max(0.0f, val + 0.5f)));
    }
  } else {
    for (int c = 0; c < channels; ++c) {
      output[out_idx_base + c] = 0;
    }
  }
}

// =============================================================================
// 7. 1D Batch Signal Convolution Kernel
// =============================================================================
__global__ void SignalConvolve1DKernel(const float* __restrict__ signal,
                                      float* __restrict__ output,
                                      size_t signal_length,
                                      const float* __restrict__ filter_kernel,
                                      size_t kernel_length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= signal_length) return;

  int radius = static_cast<int>(kernel_length / 2);
  float sum = 0.0f;

  for (size_t k = 0; k < kernel_length; ++k) {
    int sample_idx = static_cast<int>(idx) + static_cast<int>(k) - radius;
    if (sample_idx >= 0 && sample_idx < static_cast<int>(signal_length)) {
      sum += signal[sample_idx] * filter_kernel[k];
    }
  }
  output[idx] = sum;
}

// =============================================================================
// Host Launch Wrapper Implementations
// =============================================================================

float LaunchGaussianFilterNaive(const uint8_t* d_input, uint8_t* d_output,
                                int width, int height, int channels,
                                const float* h_kernel, int kernel_size,
                                cudaStream_t stream) {
  float* d_kernel = nullptr;
  size_t k_bytes = kernel_size * kernel_size * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_kernel, k_bytes));
  CUDA_CHECK(cudaMemcpyAsync(d_kernel, h_kernel, k_bytes, cudaMemcpyHostToDevice, stream));

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  GpuTimer timer;
  timer.Start(stream);
  GaussianFilterNaiveKernel<<<grid, block, 0, stream>>>(
      d_input, d_output, width, height, channels, d_kernel, kernel_size);
  timer.Stop(stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  float ms = timer.ElapsedMilliseconds();
  CUDA_CHECK(cudaFree(d_kernel));
  return ms;
}

float LaunchGaussianFilterTiled(const uint8_t* d_input, uint8_t* d_output,
                                int width, int height, int channels,
                                const float* h_kernel, int kernel_size,
                                cudaStream_t stream) {
  float* d_kernel = nullptr;
  size_t k_bytes = kernel_size * kernel_size * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_kernel, k_bytes));
  CUDA_CHECK(cudaMemcpyAsync(d_kernel, h_kernel, k_bytes, cudaMemcpyHostToDevice, stream));

  dim3 block(kSharedTileDim, kSharedTileDim);
  dim3 grid((width + kSharedTileDim - 1) / kSharedTileDim,
            (height + kSharedTileDim - 1) / kSharedTileDim);

  GpuTimer timer;
  timer.Start(stream);
  GaussianFilterTiledKernel<<<grid, block, 0, stream>>>(
      d_input, d_output, width, height, channels, d_kernel, kernel_size);
  timer.Stop(stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  float ms = timer.ElapsedMilliseconds();
  CUDA_CHECK(cudaFree(d_kernel));
  return ms;
}

float LaunchGaussianFilterConstant(const uint8_t* d_input, uint8_t* d_output,
                                   int width, int height, int channels,
                                   const float* h_kernel, int kernel_size,
                                   cudaStream_t stream) {
  size_t k_bytes = kernel_size * kernel_size * sizeof(float);
  CUDA_CHECK(cudaMemcpyToSymbolAsync(c_filter_kernel, h_kernel, k_bytes, 0,
                                     cudaMemcpyHostToDevice, stream));

  dim3 block(kSharedTileDim, kSharedTileDim);
  dim3 grid((width + kSharedTileDim - 1) / kSharedTileDim,
            (height + kSharedTileDim - 1) / kSharedTileDim);

  GpuTimer timer;
  timer.Start(stream);
  GaussianFilterConstantKernel<<<grid, block, 0, stream>>>(
      d_input, d_output, width, height, channels, kernel_size);
  timer.Stop(stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  return timer.ElapsedMilliseconds();
}

float LaunchSobelFilter(const uint8_t* d_input, uint8_t* d_output,
                        int width, int height,
                        cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  GpuTimer timer;
  timer.Start(stream);
  SobelFilterKernel<<<grid, block, 0, stream>>>(d_input, d_output, width, height);
  timer.Stop(stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  return timer.ElapsedMilliseconds();
}

float LaunchMedianFilter(const uint8_t* d_input, uint8_t* d_output,
                         int width, int height, int channels,
                         cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  GpuTimer timer;
  timer.Start(stream);
  MedianFilterKernel<<<grid, block, 0, stream>>>(d_input, d_output, width, height, channels);
  timer.Stop(stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  return timer.ElapsedMilliseconds();
}

float LaunchImageRotation(const uint8_t* d_input, uint8_t* d_output,
                          int width, int height, int channels,
                          float angle_degrees,
                          cudaStream_t stream) {
  float rad = -angle_degrees * static_cast<float>(M_PI / 180.0);
  float cos_theta = cosf(rad);
  float sin_theta = sinf(rad);
  float cx = width * 0.5f;
  float cy = height * 0.5f;

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  GpuTimer timer;
  timer.Start(stream);
  ImageRotationKernel<<<grid, block, 0, stream>>>(
      d_input, d_output, width, height, channels, cos_theta, sin_theta, cx, cy);
  timer.Stop(stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  return timer.ElapsedMilliseconds();
}

float LaunchBatchSignalConvolution1D(const float* d_signal, float* d_output,
                                    size_t signal_length,
                                    const float* d_kernel, size_t kernel_length,
                                    cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = static_cast<int>((signal_length + block_size - 1) / block_size);

  GpuTimer timer;
  timer.Start(stream);
  SignalConvolve1DKernel<<<num_blocks, block_size, 0, stream>>>(
      d_signal, d_output, signal_length, d_kernel, kernel_length);
  timer.Stop(stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  return timer.ElapsedMilliseconds();
}

}  // namespace enterprise_cuda

#endif  // __CUDACC__
