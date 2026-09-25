// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef ENTERPRISE_IMAGE_IO_HPP_
#define ENTERPRISE_IMAGE_IO_HPP_

#include <string>
#include "image_data.hpp"

namespace enterprise_cuda {

// Reads an RGB image from a PPM file (supports both binary P6 and ASCII P3 formats).
bool ReadImagePpm(const std::string& filepath, ImageRGB* out_image);

// Writes an RGB image to a binary PPM (P6) file.
bool WriteImagePpm(const std::string& filepath, const ImageRGB& image);

// Writes a Grayscale image to a PGM (P5) file.
bool WriteImagePgm(const std::string& filepath, const ImageGray& image);

// Reads a Grayscale image from a PGM file.
bool ReadImagePgm(const std::string& filepath, ImageGray* out_image);

}  // namespace enterprise_cuda

#endif  // ENTERPRISE_IMAGE_IO_HPP_
