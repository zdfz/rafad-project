# CUDA Quick Reference 🚀

**Core idea:** How to run CUDA on a GPU: launch kernels (many threads/blocks), manage device memory, use shared memory + `__syncthreads()` and avoid races.  

---

## Quick Flow & Roles 🧭
- Host = CPU + host memory (runs normal C/C++).
- Device = GPU + device memory (runs CUDA kernels).
- Typical flow: copy host → device → launch kernel → copy device → host.  
  Bottom line: CPU = boss, GPU = worker.

---

## Launching Kernels & nvcc ✨
- Kernel: `__global__` function (runs on device, called from host).
- Launch syntax: `kernel<<<blocks, threads>>>(...);` e.g. `mykernel<<<1,1>>>();`.
- `nvcc` separates host vs device compilation.

---

## Memory Management 🧠
- Host pointers: CPU-only. Device pointers: GPU-only.
- Key APIs:  
  - `cudaMalloc(void **ptr, size_t size)`  
  - `cudaMemcpy(dst, src, size, kind)`  
  - `cudaFree(ptr)`  
- Treat host/device memory as separate worlds.

---

## Blocks, Threads & Indexing (1D) 🧵
- Threads → blocks → grid.
- Built-ins: `blockIdx.x`, `threadIdx.x`, `blockDim.x`.
- 1D thread index: `int index = threadIdx.x + blockIdx.x * blockDim.x;`
- Always guard with `if (index < n)` for non-multiples.

---

## Vector Add (patterns) ➕
- One block, many threads: `c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];` → `add<<<1,N>>>(...)`.
- Many blocks, one thread: `c[blockIdx.x] = ...` → `add<<<N,1>>>(...)`.
- Blocks+threads: compute `index` then `c[index] = a[index] + b[index];`.

---

## Shared Memory & Sync 🧩
- `__shared__` = fast per-block memory.
- Threads in same block can share and sync with `__syncthreads()`.
- Example (1D stencil): load needed values (including halo) into shared array, call `__syncthreads()`, then compute. Always sync after loading to avoid data races.

---

## Asynchronicity & Errors ⏱️
- Kernel launches are asynchronous (CPU continues).
- Ways to wait: `cudaMemcpy()` (blocks), `cudaDeviceSynchronize()`, or use `cudaMemcpyAsync()` + explicit sync.
- Error checks: `cudaGetLastError()`, `cudaGetErrorString(err)`.

---

## Device Selection & Capability 🖥️
- Query: `cudaGetDeviceCount`, `cudaSetDevice`, `cudaGetDeviceProperties`.
- Compute capability (e.g., 1.x, 2.x, ...) describes GPU features (FP64, shared mem, caches).

---

## Multi-Dimensional IDs & Textures 🗺️
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim` have `.x/.y/.z`.
- Useful for 2D/3D data (images, volumes).
- Textures: read-only with caching/filtering (wrap/clamp, linear/bilinear/trilinear).

---

## Matrix Multiply (overview) 🔢
- Goal: C = A × B for N×N matrices. Row-major 1D indexing: `idx = row*cols + col`.
- Map: one thread → one output cell.
- From 1D `idx` to coords:  
  - `row = idx / N; col = idx % N;`
- Kernel pseudo-steps:
  1. compute `idx`, `if (idx < N*N)`  
  2. `row/col` from `idx`  
  3. `sum = 0; for k in 0..N-1: sum += A[row*N + k] * B[k*N + col];`  
  4. `C[idx] = sum;`
- Launch tip: `blockSize = 256; total = N*N; gridSize = (total + blockSize - 1)/blockSize;`.

---

## Why 1D arrays matter 📏
- Matrices are stored as 1D arrays; convert (row,col) ↔ linear index so threads access contiguous memory.

---

## Summary Checklist ✅
- Kernel: `__global__`, launch with `<<< >>>`.
- Indexing: `threadIdx/ blockIdx / blockDim`.
- Memory: `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
- Sync & coop: `__shared__`, `__syncthreads()`, `cudaDeviceSynchronize()`.
- Error handling: `cudaGetLastError()` & `cudaGetErrorString()`.

---

## Definitions (memorize) 📚
- Host = CPU + memory.  
- Device = GPU + memory.  
- Kernel = `__global__` GPU function.  
- Block = threads group (can share memory, sync).  
- Grid = collection of blocks.  
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim` = built-ins.  
- Shared memory = `__shared__` per-block fast memory.  
- `__syncthreads()` = per-block barrier.  
- Data race = unsynchronized concurrent read/write bug.  
- `cudaMalloc`, `cudaFree`, `cudaMemcpy` = device memory APIs.  
- Compute capability = GPU feature/version.

---
