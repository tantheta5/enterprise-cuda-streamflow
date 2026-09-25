# System Architecture & GPU Memory Hierarchy Design
## Enterprise CUDA StreamFlow Engine

---

## 1. Thread-Block Mapping & 2D Grid Topology

In image processing operations, input buffers represent two-dimensional spatial arrays of dimension $W \times H$. To maximize memory coalescing across horizontal rows of pixels and preserve spatial locality, our GPU kernels partition the problem using a 2D block and grid layout.

### Grid Dimension Configuration
For an image with width $W$ and height $H$:
- **Thread Block Dimensions**:
  $$\text{blockDim} = (16, 16) \implies 256 \text{ threads per block}$$
- **Grid Dimensions**:
  $$\text{gridDim.x} = \left\lceil \frac{W}{16} \right\rceil, \quad \text{gridDim.y} = \left\lceil \frac{H}{16} \right\rceil$$

```
  +-------------------------------------------------------------+
  | Grid: (ceil(W/16), ceil(H/16))                              |
  |  +--------------------+  +--------------------+             |
  |  | Block (0, 0)       |  | Block (1, 0)       |  ...        |
  |  | 16 x 16 threads    |  | 16 x 16 threads    |             |
  |  +--------------------+  +--------------------+             |
  |  +--------------------+  +--------------------+             |
  |  | Block (0, 1)       |  | Block (1, 1)       |  ...        |
  |  | 16 x 16 threads    |  | 16 x 16 threads    |             |
  |  +--------------------+  +--------------------+             |
  +-------------------------------------------------------------+
```

---

## 2. Memory Hierarchy Utilization

The engine targets every level of the NVIDIA GPU memory hierarchy:

```
+-------------------------------------------------------------------------+
| Level              Capacity       Latency        Bandwidth              |
+-------------------------------------------------------------------------+
| Registers (RF)     256 KB / SM    1 cycle        ~10-20 TB/s (aggregate)|
| Shared Memory (L1) 64-128 KB / SM 1-3 cycles     ~5-10 TB/s             |
| Constant Memory    64 KB (cached) 1 cycle (bcast) ~2-4 TB/s             |
| L2 Cache           4-64 MB        20-30 cycles   ~1-2 TB/s              |
| Device Global DRAM 4-80 GB        200-400 cycles 500-2000 GB/s          |
| Host Memory (PCIe) 16-256 GB      PCIe Bus       16-64 GB/s             |
+-------------------------------------------------------------------------+
```

### 2.1 Shared Memory Tiling with Halo Apron Cells
For a filter of radius $R = 2$ ($5 \times 5$ kernel), each internal pixel requires values from two rows above, two rows below, two columns left, and two columns right.

```
       <-- 2 px --> <------- 16 px -------> <-- 2 px -->
   ^   +------------+-----------------------+------------+
   |   | Top-Left   |      Top Apron        | Top-Right  |  2 rows
  2 px | Halo       |                       | Halo       |
   v   +------------+-----------------------+------------+
   ^   |            |                       |            |
   |   |            |                       |            |
 16 px | Left Apron |   Compute Core Tile   | Right Apron|  16 rows
   |   |            |   (16x16 threads)     |            |
   v   |            |                       |            |
   ^   +------------+-----------------------+------------+
  2 px | Bottom-Left|     Bottom Apron      |Bottom-Right|  2 rows
   v   | Halo       |                       | Halo       |
       +------------+-----------------------+------------+
       <------------------ 20 x 20 ---------------------->
```

- Total shared elements per block: $20 \times 20 = 400$ floats.
- Total threads in block: $16 \times 16 = 256$ threads.
- Thread stride loop: Threads $0 \dots 255$ load indices $0 \dots 255$; threads $0 \dots 143$ load indices $256 \dots 399$.
- `__syncthreads()`: Synchronizes the block so that all 400 shared elements are written before any thread begins computing.

---

## 3. Asynchronous Multi-Stream Ring Pipeline

The asynchronous multi-stream execution pipeline overlaps PCIe bus transfers with GPU kernel computation using page-locked pinned host memory:

```
Time ------------------------------------------------------------------------>
Stream 0: | HtoD (Chunk 0) | Kernel (Chunk 0) | DtoH (Chunk 0) |
Stream 1:                  | HtoD (Chunk 1)   | Kernel (Chunk 1) | DtoH (Chunk 1) |
Stream 2:                                     | HtoD (Chunk 2)   | Kernel (Chunk 2) |
```

- By keeping multiple chunks in flight, PCIe transfer latency is hidden behind compute time.
- GPU Streaming Multiprocessors remain saturated at close to 100% compute capacity.
