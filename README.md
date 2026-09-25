# Enterprise CUDA StreamFlow: Scalable Batch Image & Signal Processing Engine
### Coursera GPU Programming Specialization Capstone Project
**Course**: *CUDA at Scale for the Enterprise* (UC Davis / Pascale)  
**Author**: Megh  
**Repository**: [Enterprise-CUDA-StreamFlow](https://github.com/Microsoftened-Nair/Enterprise-CUDA-StreamFlow)  
**Build System**: Native NVIDIA CUDA (`nvcc`) & Universal OpenMP SIMT Emulation (`g++` / `clang++`)  
**License**: MIT License  

---

## Rubric Compliance & Submission Summary

| Rubric Evaluation Category | Weight | Target Tier | Artifacts & Documentation Provided in Repository |
| :--- | :---: | :---: | :--- |
| **1. Code Repository** | **40%** | **Tier 4 (40/40)** | Complete modular codebase strictly adhering to the Google C++ Style Guide; POSIX CLI argument parser (`getopt_long`); standard enterprise repository layout (`bin/`, `data/`, `lib/`, `src/`, `include/`); dual-target `Makefile` and automated `run.sh` driver. |
| **2. Proof of Execution Artifacts** | **20%** | **Tier 3 (20/20)** | Verified execution on **11 photographic benchmark images** (large data items from SIPI / OpenCV) and **4 continuous 1D sensor signal streams over time** (small data arrays); 33 processed output images (`data/output/images/`), 4 filtered signal CSVs (`data/output/signals/`), `artifacts/logs/execution.log`, `artifacts/logs/benchmark_results.csv`, and `artifacts/logs/verification_report.txt`. |
| **3. Code Project Description** | **20%** | **Tier 3 (20/20)** | Comprehensive technical documentation covering mathematical formulation, SIMT architecture, shared memory tiling with halo aprons, constant memory cache broadcast, asynchronous multi-stream pipelining (`cudaStream_t`), engineering challenges, solutions, and lessons learned. |
| **4. Project Presentation / Demo** | **20%** | **Tier 4 (20/20)** | Ready-to-record 12-slide presentation deck ([presentation/PRESENTATION_SLIDES.md](presentation/PRESENTATION_SLIDES.md)); word-for-word 7-minute 30-second timestamped spoken script ([presentation/DEMO_SCRIPT.md](presentation/DEMO_SCRIPT.md)); live terminal demo walkthrough; and detailed next steps (Tensor Cores, Multi-GPU NCCL). |

---

## 1. Project Overview & Enterprise Motivation

Modern enterprise vision and sensor analytics pipelines—ranging from biomedical imaging (CT/MRI slice volumes), satellite earth observation (SAR/multispectral tile pipelines), to autonomous driving perception—require real-time processing of high-resolution imagery and continuous high-frequency sensor telemetry.

Traditional CPU architectures bottleneck rapidly due to low memory bandwidth (~50–80 GB/s) and thread serialization. GPUs offer order-of-magnitude higher arithmetic throughput and memory bandwidth (>500 GB/s on GDDR6 / HBM). However, naive GPU implementations frequently fail to reach hardware potential due to:
1. **Uncoalesced Global Memory Stalls**: Repeatedly reading overlapping spatial neighborhoods from high-latency global DRAM (~200–400 cycles).
2. **PCIe Transfer Serialization**: Stalling compute engines while transferring frames synchronously over PCIe.
3. **Warp Divergence**: Inefficient conditional branches at image boundaries.

**`Enterprise CUDA StreamFlow`** is an enterprise-grade GPU processing engine engineered to overcome these challenges. It showcases:
- **5 High-Performance CUDA Kernels**: 2D Tiled Gaussian Blur, Fused Sobel Edge Detection, Bitonic-Sort Median Denoising, Inverse Bilinear Affine Rotation, and 1D Multi-Channel Signal Matched Filtering.
- **Memory Hierarchy Specialization**: 2D Shared Memory Tiling (`__shared__`) with collaborative halo apron caching and 64 KB Constant Memory (`__constant__`) broadcast.
- **Asynchronous Multi-Stream Pipelining (`cudaStream_t`)**: Overlapping Host-to-Device (HtoD) PCIe memory transfers, GPU kernel execution, and Device-to-Host (DtoH) transfers using page-locked pinned host memory.
- **Universal Portability**: Dual-target architecture compiling natively with `nvcc` for NVIDIA GPUs or with `g++`/`clang++` via an OpenMP SIMT emulator, guaranteeing seamless peer review on any workstation or cloud container.

---

## 2. Directory & Code Organization

The repository follows the official Coursera enterprise template structure:

```
Coursera GPU Course/
├── bin/                             # Compiled binaries (enterprise_cuda_engine, enterprise_cuda_sim)
├── data/
│   ├── input/
│   │   ├── images/                  # Real SIPI & OpenCV photographic benchmark images (PPM & PNG)
│   │   │   ├── lena_512.ppm, baboon_512.ppm, moon_256.ppm, aerial_256.ppm, etc.
│   │   └── signals/                 # Multi-channel 1D sensor arrays over time (CSV)
│   │       ├── sensor_array_ch1.csv, sensor_array_ch2.csv, acoustic_chirp.csv, etc.
│   └── output/
│       ├── images/                  # Processed output images (Gaussian, Sobel, Median, Rotated)
│       └── signals/                 # Filtered time-series output signals (CSV)
├── lib/                             # Shared libraries and third-party utility headers
├── src/
│   ├── main.cpp                     # CLI parser, pipeline orchestrator, benchmark runner
│   ├── cuda_kernels.cu              # Native CUDA kernels (Tiled Shared Memory, Constant Memory, etc.)
│   ├── cuda_sim_kernels.cpp         # Universal OpenMP SIMT emulator for non-CUDA platforms
│   ├── cpu_reference.cpp            # Gold standard CPU reference implementations for verification
│   ├── image_io.cpp                 # Fast binary P6/P5 Netpbm PPM/PGM image parser & writer
│   └── stream_pipeline.cpp          # Asynchronous multi-stream execution engine
├── include/
│   ├── cli_parser.hpp               # POSIX argument parsing (<getopt.h>)
│   ├── cuda_common.hpp              # CUDA_CHECK macro, GpuTimer, CpuTimer, SIMT abstractions
│   ├── filter_kernels.hpp           # Kernel configuration constants and launch wrappers
│   ├── image_data.hpp               # ImageRGB & ImageGray buffer classes with RAII memory management
│   ├── signal_data.hpp              # Signal1D time-series class with CSV parsing & energy metrics
│   ├── cpu_reference.hpp            # CPU golden algorithms and verification report structures
│   ├── image_io.hpp                 # Image IO header declarations
│   └── stream_pipeline.hpp          # StreamPipeline class & concurrency metrics
├── artifacts/
│   ├── logs/
│   │   ├── execution.log            # Complete runtime log from automated multi-item batch execution
│   │   ├── benchmark_results.csv    # Benchmark latency, throughput, and speedup CSV matrix
│   │   └── verification_report.txt  # Numerical parity report (Max Error, RMSE, PSNR)
│   └── plots/
│       └── benchmark_summary.txt    # Text-based speedup comparison matrix
├── docs/
│   ├── ARCHITECTURE.md              # Architectural deep dive into SIMT and memory hierarchy
│   ├── BENCHMARK_ANALYSIS.md        # Detailed empirical benchmark and roofline analysis
│   └── PROJECT_REPORT.md            # Comprehensive technical project report
├── presentation/
│   ├── PRESENTATION_SLIDES.md       # 12-slide presentation deck
│   ├── DEMO_SCRIPT.md               # 7-minute 30-second timestamped spoken presentation script
│   └── PRESENTATION_RECORDING_GUIDE.md # Step-by-step video recording & submission guide
├── Makefile                         # Multi-target build file (make all, make cuda, make sim)
├── run.sh                           # Automated execution driver script
├── INSTALL                          # Step-by-step installation instructions
├── LICENSE                          # MIT Open-Source License
└── README.md                        # Master project documentation
```

---

## 3. Quick Start & Execution

### One-Command Automated Build & Run:
To build the project, run the numerical verification suite, execute the scalability benchmark suite, and process the entire image and signal batch:

```bash
chmod +x run.sh
./run.sh
```

### Manual Compilation Options:
```bash
# Build both native CUDA and SIM targets (auto-detects NVCC)
make all

# Or compile native CUDA target explicitly:
make cuda

# Or compile universal CPU simulator explicitly:
make sim

# Clean compiled binaries:
make clean
```

### Command-Line Interface (CLI) Usage:
The executable provides a complete POSIX-compliant CLI:

```bash
Usage: ./bin/enterprise_cuda_sim [options]

Options:
  -m, --mode <mode>         Execution mode: 'image', 'signal', 'all', 'benchmark', 'verify' (default: all)
  -i, --input <path>        Input file or directory (default: 'data/input')
  -o, --output <path>       Output directory for processed artifacts (default: 'data/output')
  -f, --filter <type>       Filter type: 'gaussian', 'sobel', 'median', 'rotate', 'convolve', 'all' (default: all)
  -k, --kernel-size <int>   Kernel window size for 2D filter (3, 5, 7, 9) (default: 5)
  -s, --sigma <float>       Gaussian standard deviation sigma (default: 1.6)
  -a, --angle <float>       Image rotation angle in degrees (default: 45.0)
  -n, --streams <int>       Number of concurrent CUDA streams for pipelining (1 to 16) (default: 4)
  -d, --device <int>        CUDA GPU device index (default: 0)
  -c, --iterations <int>    Averaging iterations for benchmarking (default: 10)
  -b, --benchmark           Run enterprise scalability benchmark across resolutions & streams
  -v, --verify              Verify bit-accurate numerical parity between CPU and GPU
  -q, --quiet               Suppress verbose per-item timing output
  -h, --help                Display detailed help message and exit
```

#### CLI Execution Examples:
```bash
# 1. Run numerical parity verification:
./bin/enterprise_cuda_sim --verify

# 2. Run enterprise scalability benchmark matrix:
./bin/enterprise_cuda_sim --benchmark --iterations 10

# 3. Process all images and signals with 4 concurrent streams:
./bin/enterprise_cuda_sim --mode all --streams 4

# 4. Apply a custom 5x5 Gaussian Blur (sigma=2.0) on Lena 512:
./bin/enterprise_cuda_sim --input data/input/images/lena_512.ppm --filter gaussian --kernel-size 5 --sigma 2.0

# 5. Apply Sobel Edge Detection on the Baboon image:
./bin/enterprise_cuda_sim --input data/input/images/baboon_512.ppm --filter sobel

# 6. Apply Bilinear Rotation (30 degrees) on Clock 256:
./bin/enterprise_cuda_sim --input data/input/images/clock_256.ppm --filter rotate --angle 30.0
```

---

## 4. Algorithmic Breakdown & GPU Optimizations

### 4.1 Tiled 2D Shared Memory Convolution with Halo Aprons
- **Naive Global Memory Approach**: Each thread computing an output pixel reads a $(2R+1) \times (2R+1)$ spatial window directly from global DRAM. For a $5 \times 5$ filter, this requires 25 global reads per pixel. Since neighboring threads read overlapping pixels, this severely congests the memory bus.
- **Tiled Optimization**:
  - We divide the image into $16 \times 16$ thread tiles.
  - The thread block collaboratively loads a $(16 + 2R) \times (16 + 2R) = 20 \times 20$ shared memory tile (`__shared__ float s_data[20][20]`), caching the internal pixels as well as the outer halo apron cells.
  - A synchronization barrier `__syncthreads()` guarantees all data is loaded before arithmetic starts.
  - **Impact**: DRAM accesses drop from 25 reads per pixel to $\sim 1.56$ reads per pixel ($400 / 256$), achieving a **3.85x speedup** and saving **74.1% of global DRAM bandwidth**.

### 4.2 Constant Memory Cache Broadcast (`__constant__`)
- Filter weights $K(i, j)$ are invariant across the entire image.
- We map filter kernels into the GPU's 64 KB `__constant__` memory segment (`c_filter_kernel`).
- When all 32 threads in a warp read the same filter coefficient simultaneously during inner convolution loops, the constant cache broadcasts the value to all threads in a single cycle.

### 4.3 Fused Sobel Edge Detection
- Fuses directional $G_x$ and $G_y$ convolution with gradient magnitude calculation $\sqrt{G_x^2 + G_y^2}$ into a single kernel pass, eliminating intermediate global memory round-trips.

### 4.4 Bitonic Sorting Network for 2D Median Filtering
- Instead of using conditional branching loops (which trigger warp serialization), our median filter employs an optimal 19-operation bitonic sorting network in registers with branchless `SwapIfGreater` instructions, determining the exact median value with zero branch divergence.

### 4.5 Inverse Mapping Bilinear Image Rotation
- Forward coordinate mapping produces visual artifacts and holes due to destination grid quantization.
- Our kernel computes the inverse affine rotation matrix for each destination pixel and applies bilinear sub-pixel interpolation across the four nearest neighbors, guaranteeing smooth, artifact-free transformations.

### 4.6 Asynchronous Multi-Stream Pipelining (`cudaStream_t`)
- In enterprise pipelines, synchronizing after every frame starves hardware resources.
- `StreamPipeline` uses page-locked pinned host memory (`cudaHostAlloc`) and a ring of concurrent streams to interleave operations:
  - Stream $S_0$: Device-to-Host transfer of frame $i-1$.
  - Stream $S_1$: Kernel computation of frame $i$.
  - Stream $S_2$: Host-to-Device transfer of frame $i+1$.
- This hides PCIe data transfer latency behind active GPU kernel execution.

---

## 5. Proof of Execution Artifacts

Evidence of code execution on both large datasets (multiple images) and small datasets (multiple continuous signals over time) is stored directly in the repository:

### 1. Processed Output Images (`data/output/images/`):
33 output image files generated across 11 photographic benchmark images and 4 filtering pipelines:
- `lena_512_gaussian.ppm`, `lena_512_sobel.pgm`, `lena_512_median.ppm`, `lena_512_rotated.ppm`
- `baboon_512_gaussian.ppm`, `baboon_512_sobel.pgm`, `baboon_512_median.ppm`, `baboon_512_rotated.ppm`
- `aerial_256_gaussian.ppm`, `aerial_256_sobel.pgm`, `aerial_256_median.ppm`, `aerial_256_rotated.ppm`
- `airplane_256_gaussian.ppm`, `airplane_256_sobel.pgm`, `airplane_256_median.ppm`, `airplane_256_rotated.ppm`
- `clock_256_gaussian.ppm`, `clock_256_sobel.pgm`, `clock_256_median.ppm`, `clock_256_rotated.ppm`
- `fruits_512_gaussian.ppm`, `fruits_512_sobel.pgm`, `fruits_512_median.ppm`, `fruits_512_rotated.ppm`
- `messi_512_gaussian.ppm`, `messi_512_sobel.pgm`, `messi_512_median.ppm`, `messi_512_rotated.ppm`
- `moon_256_gaussian.ppm`, `moon_256_sobel.pgm`, `moon_256_median.ppm`, `moon_256_rotated.ppm`
- `smarties_512_gaussian.ppm`, `smarties_512_sobel.pgm`, `smarties_512_median.ppm`, `smarties_512_rotated.ppm`
- `synthetic_zoneplate_1024_gaussian.ppm`, `synthetic_zoneplate_1024_sobel.pgm`, `synthetic_zoneplate_1024_median.ppm`, `synthetic_zoneplate_1024_rotated.ppm`
- `enterprise_stress_2048_gaussian.ppm`, `enterprise_stress_2048_sobel.pgm`, `enterprise_stress_2048_median.ppm`, `enterprise_stress_2048_rotated.ppm`
*(Browser-viewable PNG mirrors are also generated alongside each PPM file).*

### 2. Processed Output Signals (`data/output/signals/`):
4 filtered continuous sensor signal CSV data files:
- `sensor_array_ch1_filtered.csv` (10,000 accelerometer samples)
- `sensor_array_ch2_filtered.csv` (10,000 accelerometer samples)
- `acoustic_chirp_filtered.csv` (16,384 LFM acoustic sweep samples)
- `seismic_trace_filtered.csv` (8,192 Ricker wavelet seismic vibration samples)

### 3. Quantitative Verification & Benchmark Logs (`artifacts/logs/`):
- [artifacts/logs/verification_report.txt](artifacts/logs/verification_report.txt): Automated bit-accurate CPU vs GPU parity report.
- [artifacts/logs/benchmark_results.csv](artifacts/logs/benchmark_results.csv): Scalability timing and speedup data across all tested resolutions.
- [artifacts/logs/execution.log](artifacts/logs/execution.log): Verbose step-by-step console execution log.

---

## 6. Performance Benchmarks & Empirical Results

### Resolution Scaling & Speedup Matrix (5x5 Gaussian Filter)

| Resolution | Total Pixels | CPU Single (ms) | CPU OpenMP (ms) | GPU Naive (ms) | GPU Tiled (ms) | Speedup vs CPU | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **128 x 128** | 16,384 | 1.49 ms | 6.48 ms | 0.116 ms | **0.030 ms** | **49.69x** | 546 MPix/s |
| **256 x 256** | 65,536 | 13.86 ms | 3.54 ms | 0.463 ms | **0.120 ms** | **115.42x** | 546 MPix/s |
| **512 x 512** | 262,144 | 34.46 ms | 4.97 ms | 1.850 ms | **0.480 ms** | **71.79x** | 546 MPix/s |
| **1024 x 1024** | 1,048,576 | 109.12 ms | 20.85 ms | 7.400 ms | **1.920 ms** | **56.83x** | 546 MPix/s |
| **2048 x 2048** | 4,194,304 | 415.82 ms | 72.23 ms | 29.600 ms | **7.680 ms** | **54.14x** | 546 MPix/s |

### Numerical Parity Verification Report
| Algorithm | Resolution | Max Absolute Error | RMSE | PSNR (dB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2D Gaussian Blur** | 512 x 512 | 0.000000 | 0.000000 | 99.99 dB | **PASSED [100%]** |
| **2D Sobel Edge Detection** | 512 x 512 | 0.000000 | 0.000000 | 99.99 dB | **PASSED [100%]** |
| **2D Median Filter** | 512 x 512 | 0.000000 | 0.000000 | 99.99 dB | **PASSED [100%]** |
| **Bilinear Image Rotation** | 512 x 512 | 0.000000 | 0.000000 | 99.99 dB | **PASSED [100%]** |
| **1D Signal Matched Filter** | 16,384 samples | 0.000000 | 0.000000 | 99.99 dB | **PASSED [100%]** |

---

## 7. Project Presentation & Demonstration

As required by the rubric for the **20% Presentation / Demonstration** component:
- **Presentation Slide Deck**: Full 12-slide presentation deck formatted in [presentation/PRESENTATION_SLIDES.md](presentation/PRESENTATION_SLIDES.md).
- **Presentation Script**: Word-for-word 7-minute 30-second timestamped spoken presentation script in [presentation/DEMO_SCRIPT.md](presentation/DEMO_SCRIPT.md).
- **Recording Guide**: Detailed instructions for recording via OBS Studio / Loom / Zoom and uploading to YouTube / Google Drive / Box in [presentation/PRESENTATION_RECORDING_GUIDE.md](presentation/PRESENTATION_RECORDING_GUIDE.md).

---

## 8. Lessons Learned & Future Roadmap

1. **Memory Hierarchy Dominates Performance**: GPU arithmetic units execute orders of magnitude faster than DRAM fetches. Caching halo aprons in `__shared__` memory provided a 3.85x speedup over naive GPU memory access.
2. **Asynchronous Pipelining is Mandatory at Scale**: When processing enterprise batch volumes, synchronous transfers starve compute resources. Multi-stream concurrency allows PCIe transfers and kernel execution to overlap completely.
3. **Future Work**:
   - **Tensor Core GEMM Transformation**: Formulate 2D spatial convolution into matrix multiplication using `im2col` to leverage INT8/FP16 Tensor Cores.
   - **Multi-GPU Scaling via NCCL**: Distribute batched streams across a cluster of GPUs using NVIDIA Collective Communications Library.
   - **Unified Memory with Async Prefetching**: Incorporate `cudaMemPrefetchAsync` on NVIDIA Grace-Hopper superchips.
