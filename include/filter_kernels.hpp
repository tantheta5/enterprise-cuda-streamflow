// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef ENTERPRISE_FILTER_KERNELS_HPP_
#define ENTERPRISE_FILTER_KERNELS_HPP_

#include <cstdint>
#include <vector>
#include "cuda_common.hpp"

namespace enterprise_cuda {

// Execution configuration constants
constexpr int kTileDim = 16;
constexpr int kMaxKernelRadius = 4; // supports up to 9x9 kernels
constexpr int kMaxKernelSize = 2 * kMaxKernelRadius + 1;

// Constant memory kernel weight buffer limit
constexpr int kMaxConstantKernelElements = kMaxKernelSize * kMaxKernelSize;

// -----------------------------------------------------------------------------
// Host Launch Wrapper Interfaces
// -----------------------------------------------------------------------------

// Launch Naive 2D Gaussian Blur (Global Memory accesses only)
float LaunchGaussianFilterNaive(const uint8_t* d_input, uint8_t* d_output,
                                int width, int height, int channels,
                                const float* h_kernel, int kernel_size,
                                cudaStream_t stream = 0);

// Launch Optimized 2D Gaussian Blur (Tiled Shared Memory with Apron)
float LaunchGaussianFilterTiled(const uint8_t* d_input, uint8_t* d_output,
                                int width, int height, int channels,
                                const float* h_kernel, int kernel_size,
                                cudaStream_t stream = 0);

// Launch Constant-Memory Accelerated 2D Gaussian Blur
float LaunchGaussianFilterConstant(const uint8_t* d_input, uint8_t* d_output,
                                   int width, int height, int channels,
                                   const float* h_kernel, int kernel_size,
                                   cudaStream_t stream = 0);

// Launch 2D Sobel Filter with Fused Magnitude
float LaunchSobelFilter(const uint8_t* d_input, uint8_t* d_output,
                        int width, int height,
                        cudaStream_t stream = 0);

// Launch 2D Median Filter (3x3 sorting network)
float LaunchMedianFilter(const uint8_t* d_input, uint8_t* d_output,
                         int width, int height, int channels,
                         cudaStream_t stream = 0);

// Launch Bilinear Image Rotation
float LaunchImageRotation(const uint8_t* d_input, uint8_t* d_output,
                          int width, int height, int channels,
                          float angle_degrees,
                          cudaStream_t stream = 0);

// Launch 1D Batch Signal Convolution
float LaunchBatchSignalConvolution1D(const float* d_signal, float* d_output,
                                    size_t signal_length,
                                    const float* d_kernel, size_t kernel_length,
                                    cudaStream_t stream = 0);

}  // namespace enterprise_cuda

#endif  // ENTERPRISE_FILTER_KERNELS_HPP_
