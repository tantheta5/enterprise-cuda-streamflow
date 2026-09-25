// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef ENTERPRISE_CPU_REFERENCE_HPP_
#define ENTERPRISE_CPU_REFERENCE_HPP_

#include <vector>
#include "image_data.hpp"
#include "signal_data.hpp"

namespace enterprise_cuda {

// Computes 2D Gaussian kernel weights normalized to sum to 1.0.
std::vector<float> GenerateGaussianKernel2D(int kernel_size, float sigma);

// CPU Gold-standard reference: 2D Gaussian Blur (RGB).
void CpuGaussianBlurRgb(const ImageRGB& input, ImageRGB* output, int kernel_size, float sigma);

// CPU Gold-standard reference: 2D Sobel Edge Detection (Grayscale).
void CpuSobelFilter(const ImageGray& input, ImageGray* output);

// CPU Gold-standard reference: 2D Median Filter (Grayscale / RGB).
void CpuMedianFilterRgb(const ImageRGB& input, ImageRGB* output, int window_size);

// CPU Gold-standard reference: Bilinear Image Rotation.
void CpuRotateImageRgb(const ImageRGB& input, ImageRGB* output, float angle_degrees);

// CPU Gold-standard reference: 1D Signal Convolution.
void CpuConvolve1D(const Signal1D& input, const Signal1D& filter_kernel, Signal1D* output);

// Verification metrics struct
struct VerificationReport {
  double max_absolute_error = 0.0;
  double mean_squared_error = 0.0;
  double root_mean_squared_error = 0.0;
  double psnr_db = 0.0;
  bool passed = true;
};

// Compares two RGB images and computes error metrics.
VerificationReport CompareImages(const ImageRGB& ref, const ImageRGB& test, double tolerance = 1.0);

// Compares two Grayscale images.
VerificationReport CompareImages(const ImageGray& ref, const ImageGray& test, double tolerance = 1.0);

// Compares two 1D signals.
VerificationReport CompareSignals(const Signal1D& ref, const Signal1D& test, double tolerance = 1e-4);

}  // namespace enterprise_cuda

#endif  // ENTERPRISE_CPU_REFERENCE_HPP_
