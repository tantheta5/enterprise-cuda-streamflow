// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include "image_io.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace enterprise_cuda {

namespace {

// Helper to consume whitespace and comment lines in PPM headers.
void SkipPpmComments(std::ifstream& infile) {
  while (infile >> std::ws && infile.peek() == '#') {
    std::string comment_line;
    std::getline(infile, comment_line);
  }
}

}  // namespace

bool ReadImagePpm(const std::string& filepath, ImageRGB* out_image) {
  if (out_image == nullptr) return false;

  std::ifstream infile(filepath, std::ios::binary);
  if (!infile.is_open()) {
    std::fprintf(stderr, "[ImageIO] Error: Unable to open file '%s' for reading\n", filepath.c_str());
    return false;
  }

  std::string magic_number;
  infile >> magic_number;
  if (magic_number != "P6" && magic_number != "P3") {
    std::fprintf(stderr, "[ImageIO] Error: Unsupported PPM format '%s' in '%s'. Must be P6 or P3.\n",
                 magic_number.c_str(), filepath.c_str());
    return false;
  }

  SkipPpmComments(infile);
  int width = 0;
  infile >> width;

  SkipPpmComments(infile);
  int height = 0;
  infile >> height;

  SkipPpmComments(infile);
  int max_val = 0;
  infile >> max_val;

  if (width <= 0 || height <= 0 || max_val <= 0) {
    std::fprintf(stderr, "[ImageIO] Error: Invalid PPM header parameters in '%s' (W=%d, H=%d, Max=%d)\n",
                 filepath.c_str(), width, height, max_val);
    return false;
  }

  // Consume the single whitespace/newline following max_val
  infile.get();

  out_image->Allocate(width, height);
  uint8_t* dst = out_image->data();
  size_t total_bytes = out_image->SizeInBytes();

  if (magic_number == "P6") {
    // Binary read
    infile.read(reinterpret_cast<char*>(dst), total_bytes);
    if (!infile) {
      std::fprintf(stderr, "[ImageIO] Warning: Incomplete binary read for '%s'\n", filepath.c_str());
    }
  } else {
    // ASCII P3 read
    int r, g, b;
    size_t idx = 0;
    while (infile >> r >> g >> b && idx < total_bytes) {
      dst[idx++] = static_cast<uint8_t>(r);
      dst[idx++] = static_cast<uint8_t>(g);
      dst[idx++] = static_cast<uint8_t>(b);
    }
  }

  return true;
}

bool WriteImagePpm(const std::string& filepath, const ImageRGB& image) {
  std::ofstream outfile(filepath, std::ios::binary);
  if (!outfile.is_open()) {
    std::fprintf(stderr, "[ImageIO] Error: Unable to create output PPM '%s'\n", filepath.c_str());
    return false;
  }

  // Write P6 binary header
  outfile << "P6\n";
  outfile << "# Created by Enterprise CUDA StreamFlow Engine\n";
  outfile << image.width() << " " << image.height() << "\n";
  outfile << "255\n";

  outfile.write(reinterpret_cast<const char*>(image.data()), image.SizeInBytes());
  outfile.flush();
  return outfile.good();
}

bool WriteImagePgm(const std::string& filepath, const ImageGray& image) {
  std::ofstream outfile(filepath, std::ios::binary);
  if (!outfile.is_open()) {
    std::fprintf(stderr, "[ImageIO] Error: Unable to create output PGM '%s'\n", filepath.c_str());
    return false;
  }

  // Write P5 binary grayscale header
  outfile << "P5\n";
  outfile << "# Created by Enterprise CUDA StreamFlow Engine\n";
  outfile << image.width() << " " << image.height() << "\n";
  outfile << "255\n";

  outfile.write(reinterpret_cast<const char*>(image.data()), image.SizeInBytes());
  outfile.flush();
  return outfile.good();
}

bool ReadImagePgm(const std::string& filepath, ImageGray* out_image) {
  if (out_image == nullptr) return false;

  std::ifstream infile(filepath, std::ios::binary);
  if (!infile.is_open()) {
    std::fprintf(stderr, "[ImageIO] Error: Unable to open PGM file '%s'\n", filepath.c_str());
    return false;
  }

  std::string magic_number;
  infile >> magic_number;
  if (magic_number != "P5" && magic_number != "P2") {
    std::fprintf(stderr, "[ImageIO] Error: Unsupported PGM format '%s'. Must be P5 or P2.\n",
                 magic_number.c_str());
    return false;
  }

  SkipPpmComments(infile);
  int width = 0;
  infile >> width;

  SkipPpmComments(infile);
  int height = 0;
  infile >> height;

  SkipPpmComments(infile);
  int max_val = 0;
  infile >> max_val;

  infile.get(); // consume separator

  out_image->Allocate(width, height);
  if (magic_number == "P5") {
    infile.read(reinterpret_cast<char*>(out_image->data()), out_image->SizeInBytes());
  } else {
    int val;
    size_t idx = 0;
    while (infile >> val && idx < out_image->SizeInBytes()) {
      out_image->data()[idx++] = static_cast<uint8_t>(val);
    }
  }

  return true;
}

}  // namespace enterprise_cuda
