**Core idea:** These slides teach how to run **CUDA** code on a GPU: launching **kernels** with many **threads/blocks**, managing **device memory**, using **shared memory** and **__syncthreads()**, and mapping problems like vector and matrix multiplication onto threads.[1]

***

## Core Concepts (Must-Know)

### Host vs Device & Basic Flow

- **Host** = CPU and its **host memory**; runs normal C/C++ code.[1]
- **Device** = GPU and its **device memory**; runs CUDA **kernels**.[1]
- Basic flow is like sending work to a helper:  
  - Copy input from **host memory** to **device memory**.[1]
  - Run GPU **kernel** on data (many threads).[1]
  - Copy results back from device to host.[1]

**Bottom Line:** Think “CPU = boss, GPU = worker”; CPU sends data + kernel, GPU does massive parallel work, then CPU collects results.[1]

***

### Kernels, __global__, and <<< >>> Launch

- A **kernel** is a GPU function marked with **__global__**.[1]
  - **Runs on device**, **called from host**.[1]
- You launch a kernel with **<<<blocks, threads>>>**, for example:  
  - `mykernel<<<1,1>>>();` or `add<<<N,1>>>(...);` or `add<<<N,M>>>(...);`.[1]
- The compiler **nvcc** splits host code (to gcc/clang) and device code (to NVIDIA compiler).[1]

**Bottom Line:** A kernel is just a special function that runs on the GPU, and `<<<...>>>` is the magic syntax to launch many parallel copies of it.[1]

***

### Device Memory Management

- **Host pointers** point to CPU memory and **cannot** be dereferenced on the GPU.[1]
- **Device pointers** point to GPU memory and **cannot** be dereferenced on the CPU.[1]
- Key functions (similar to malloc/free/memcpy):  
  - **cudaMalloc(void **ptr, size_t size)**: allocate device memory.[1]
  - **cudaMemcpy(dst, src, size, kind)**: copy between host/device.[1]
  - **cudaFree(ptr)**: free device memory.[1]

**Bottom Line:** Treat host and device memory as two separate worlds, and always use cudaMalloc / cudaMemcpy / cudaFree to move data between them.[1]

***

### Blocks, Threads, and Indexing (1D)

- CUDA runs many copies of a kernel in parallel; each copy is a **thread**, grouped into **blocks**, blocks form a **grid**.[1]
- Built-in variables:  
  - **blockIdx.x** = which block in the grid.[1]
  - **threadIdx.x** = which thread inside the block.[1]
  - **blockDim.x** = number of threads per block.[1]
- A unique 1D index per thread (for vector/array) is:  
  - `int index = threadIdx.x + blockIdx.x * blockDim.x;`.[1]

**Bottom Line:** Each thread needs a unique index so it knows “which element is mine”; combine block and thread indices to get that index.[1]

***

### Vector Addition Examples

- **One block, many threads:**  
  - Kernel: `c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];`.[1]
  - Launch: `add<<<1,N>>>(d_a, d_b, d_c);`.[1]
- **Many blocks, one thread each:**  
  - Kernel: `c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];`.[1]
  - Launch: `add<<<N,1>>>(d_a, d_b, d_c);`.[1]
- **Blocks + threads:**  
  - Kernel uses `index = threadIdx.x + blockIdx.x * blockDim.x;` then `c[index] = a[index] + b[index];`.[1]
  - Launch: `add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(...);`.[1]

**Bottom Line:** Vector add is the “hello world” of CUDA: every thread just adds one pair of elements and writes one result.[1]

***

### Handling Arbitrary N (Bounds Check)

- Real **N** is often not a clean multiple of **blockDim.x**.[1]
- Launch trick: `add<<<(N + M - 1)/M, M>>>(...);` for M threads per block.[1]
- Inside kernel: check `if (index < n)` before accessing arrays.[1]

**Bottom Line:** Always guard memory access with `if (index < n)` when your launch may produce extra threads.[1]

***

### Why Threads (Not Only Blocks)

- Blocks are **independent**; they cannot directly talk to each other.[1]
- Threads inside the **same block** can:  
  - Communicate via **shared memory**.[1]
  - Synchronize with **__syncthreads()**.[1]

**Bottom Line:** Threads inside a block are like teammates in one room who can share a whiteboard (shared memory), while blocks are like separate rooms that cannot talk.[1]

***

### Shared Memory & __syncthreads() with 1D Stencil

- **Shared memory** (declared with **__shared__**) is fast on-chip memory visible to threads in the same block only.[1]
- In the 1D stencil:  
  - Each thread handles one output element.[1]
  - All threads copy needed input into **shared** array `temp`.[1]
  - Extra elements at edges are called **halo**.[1]
- **__syncthreads()** is a barrier: all threads in the block wait until everyone reaches that line.[1]

**Bottom Line:** Shared memory + __syncthreads() turn many threads into a coordinated team that first loads data together, then safely computes.[1]

***

### Data Race and Fix in Stencil

- Without sync, a thread may read **temp[...]** before another thread has written its halo values.[1]
- This is a **data race**: order of reads/writes is unpredictable.[1]
- Solution: after loading shared memory, call **__syncthreads()** before using `temp`.[1]

**Bottom Line:** Always synchronize threads (with __syncthreads) after shared loading and before using shared data, or results will be random and buggy.[1]

***

### Managing Device Execution & Errors

- **Kernel launches are asynchronous**: CPU continues immediately; GPU runs in background.[1]
- Ways to synchronize CPU with GPU:  
  - **cudaMemcpy()**: blocks until copy + previous CUDA calls are done.[1]
  - **cudaMemcpyAsync()**: non-blocking; combine with **cudaDeviceSynchronize()** if you need to wait.[1]
- Error handling functions:  
  - **cudaGetLastError()** → last error.[1]
  - **cudaGetErrorString(err)** → printable description.[1]

**Bottom Line:** The GPU runs independently; use memcpy/sync calls and error-checking to be sure work really finished and succeeded.[1]

***

### Device Selection & Compute Capability

- You can query and choose GPUs:  
  - **cudaGetDeviceCount**, **cudaSetDevice**, **cudaGetDevice**, **cudaGetDeviceProperties**.[1]
- **Compute capability** describes GPU architecture features (registers, memory sizes, double precision, caches, etc.).[1]
- Different compute capabilities (1.x, 2.x, etc.) support different features like double precision or concurrent kernels.[1]

**Bottom Line:** Not all GPUs are equal; use device queries and compute capability to know what your code can safely use.[1]

***

### Multi-Dimensional IDs and Textures (Overview)

- **threadIdx**, **blockIdx**, **blockDim**, **gridDim** can be 1D, 2D, or 3D: `.x`, `.y`, `.z`.[1]
- This helps map naturally to 2D/3D data like images and volumes.[1]
- **Textures**:  
  - Read-only objects with special **caches** and **filtering** (linear, bilinear, trilinear).[1]
  - Handle out-of-bounds via wrap/clamp, and can be 1D/2D/3D.[1]

**Bottom Line:** CUDA’s multi-dimensional indices and textures make it easier and faster to handle images and 3D data.[1]

***

### CUDA Summary (From Slides)

- To write CUDA programs you must know:  
  - **__global__**, **blockIdx.x**, **threadIdx.x**, `<<< >>>`.[1]
  - **cudaMalloc**, **cudaMemcpy**, **cudaFree** for memory.[1]
  - **__shared__**, **__syncthreads**, **cudaMemcpy vs cudaMemcpyAsync**, **cudaDeviceSynchronize**.[1]

**Bottom Line:** CUDA basics = how to launch kernels, index threads, manage GPU memory, and synchronize/communicate efficiently.[1]

***

### Matrix Multiplication on CPU vs GPU

- Goal: given **two N×N matrices** X and Y, compute result matrix **ans** = X × Y.[1]
- Math rule: each output **C[i][j] = sum over k of A[i][k] * B[k][j]**.[1]
- 2D → 1D index formula (row-major):  
  - `index(row, col) = row * cols + col`.[1]
  - So `A[i][k] = A[i*CA + k]`, `B[k][j] = B[k*CB + j]`, `C[i][j] = C[i*CB + j]`.[1]

**Bottom Line:** Matrix multiply builds each cell from a row and a column, and on GPU you usually store and index matrices as simple 1D arrays.[1]

***

### Why 1D Arrays Matter

- Even 2D matrices are stored as one long array in memory like `[a00, a01, a02, a10, a11, a12]`.[1]
- The 1D index lets CUDA threads work easily since each thread often uses one linear index `idx`.[1]

**Bottom Line:** GPUs love 1D arrays; you convert (row, col) to a single index so each thread can work with a contiguous memory model.[1]

***

### CUDA Execution Model for Matrix Multiply

- A **kernel launch** creates many threads; each thread runs the **same** kernel code but on **different data**.[1]
- Simplest mapping for N×N:  
  - **N*N threads**, each computing exactly **one** output cell.[1]
- Each thread has a unique linear id **idx**.[1]

**Bottom Line:** Matrix multiply on GPU is “one thread = one output cell,” so all cells are computed in parallel.[1]

***

### Mapping 1D idx → (row, col)

- Compute `idx = blockIdx.x * blockDim.x + threadIdx.x;`.[1]
- Convert to coordinates:  
  - `row = idx / N;` (integer division).[1]
  - `col = idx % N;`.[1]
- Then compute that cell:  
  - `C[idx] = sum_k A[row*N + k] * B[k*N + col];`.[1]

**Bottom Line:** Use division and modulo to turn a flat thread index into row and column positions in the matrix.[1]

***

### Matrix Multiply Kernel (1D Style)

- Example kernel:  
  - Compute `idx`, `total = N*N`, and check `if (idx < total)`.[1]
  - Derive `row` and `col` from `idx`.[1]
  - Loop `k` from 0 to N-1, accumulate `sum += A[row*N + k] * B[k*N + col];`.[1]
  - Store `C[idx] = sum;`.[1]
- Each thread only reads its needed row of A and column of B and writes a single C element.[1]

**Bottom Line:** Inside the kernel, each thread just runs a tiny for-loop over k to compute its assigned C cell.[1]

***

### Example of What One Thread Touches

- For N=4 and `idx=6` → `(row=1, col=2)`.[1]
- It reads: `A[4], A[5], A[6], A[7]` and `B[2], B[6], B[10], B[14]`, then writes `C[6]`.[1]
- So one thread touches a small slice of A and B plus one output spot.[1]

**Bottom Line:** Each thread’s footprint is small and independent, which is why GPU can run many threads safely in parallel.[1]

***

### Boundary Check and Launch Configuration

- Need `if (idx < N*N)` because often you launch **more threads than cells** (due to block size).[1]
- Pick:  
  - `blockSize = 256;` (common).[1]
  - `total = N*N;`.[1]
  - `gridSize = (total + blockSize - 1) / blockSize;`.[1]
- Total threads = `gridSize * blockSize`, which covers at least `N*N`.[1]

**Bottom Line:** Use grid/block math to cover the full matrix, then protect against extra threads with a simple if-condition.[1]

***

### CPU vs GPU View

- **CPU version:** one worker does C, then C, then C, all the way to C[N*N-1].[1]
- **GPU version:** many workers; each thread computes **one C[idx]** at the same time.[1]

**Bottom Line:** GPU turns a long serial matrix multiply loop into a huge set of independent small tasks done in parallel.[1]

***

## Supporting Details (Likely to Be Tested)

- **Hello World CUDA:** you can compile plain C code with `nvcc` even without device code, and then extend it by adding a trivial kernel and launch.[1]
- The add-on device example allocates device memory, copies inputs, launches `add<<<1,1>>>`, copies result back, and frees memory.[1]
- **Reviews** in slides highlight:  
  - Host vs device roles.[1]
  - Use of CUDA keywords and APIs (`__global__`, `blockIdx.x`, `threadIdx.x`, `cudaMalloc`, `cudaMemcpy`, `cudaFree`).[1]
  - Use of shared memory and __syncthreads for safe cooperation.[1]

**Bottom Line:** Exam questions will likely poke your memory of the standard CUDA pattern: allocate–copy–launch–sync–copy back–free.[1]

***

## Definitions (Memorize-Friendly)

- **Host:** CPU and its memory running normal C/C++ code.[1]
- **Device:** GPU and its memory running CUDA kernels.[1]
- **Kernel:** A function marked with `__global__` that runs on the GPU and is launched from the CPU.[1]
- **Block:** A group of threads that can share **shared memory** and use **__syncthreads()**.[1]
- **Grid:** Collection of blocks launched by one kernel call.[1]
- **threadIdx, blockIdx, blockDim, gridDim:** Built-in variables that give thread/block positions and dimensions (can be 1D/2D/3D).[1]
- **Shared memory:** Fast on-chip memory shared only by threads in the same block, declared with `__shared__`.[1]
- **__syncthreads():** Barrier that makes all threads in a block wait until everyone reaches that line.[1]
- **Data race:** Bug where threads read/write the same data without proper order or sync, leading to unpredictable results.[1]
- **cudaMalloc / cudaFree / cudaMemcpy:** CUDA API calls for allocating, freeing, and copying device memory.[1]
- **Compute capability:** Version number describing GPU architecture features (memory sizes, instructions, etc.).[1]

**Bottom Line:** Knowing these short definitions lets you quickly parse exam questions and code snippets.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/44255112/b764d3b9-1127-4ec4-83ac-cd9a0baebbce/13-GPU-Programming-CUDA-2025-2026.pdf)
