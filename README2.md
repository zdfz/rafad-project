**Core idea:** These slides teach how to run **CUDA** code on a GPU: launching **kernels** with many **threads/blocks**, managing **device memory**, using **shared memory** and **__syncthreads()**, and mapping problems like vector and matrix multiplication onto threads.

***

## Core Concepts (Must-Know)

### Host vs Device & Basic Flow

- **Host** = CPU and its **host memory**; runs normal C/C++ code.
- **Device** = GPU and its **device memory**; runs CUDA **kernels**.
- Basic flow is like sending work to a helper:  
  - Copy input from **host memory** to **device memory**.
  - Run GPU **kernel** on data (many threads).
  - Copy results back from device to host.

**Bottom Line:** Think “CPU = boss, GPU = worker”; CPU sends data + kernel, GPU does massive parallel work, then CPU collects results.

***

### Kernels, __global__, and <<< >>> Launch

- A **kernel** is a GPU function marked with **__global__**.
  - **Runs on device**, **called from host**.
- You launch a kernel with **<<<blocks, threads>>>**, for example:  
  - `mykernel<<<1,1>>>();` or `add<<<N,1>>>(...);` or `add<<<N,M>>>(...);`.
- The compiler **nvcc** splits host code (to gcc/clang) and device code (to NVIDIA compiler).

**Bottom Line:** A kernel is just a special function that runs on the GPU, and `<<<...>>>` is the magic syntax to launch many parallel copies of it.

***

### Device Memory Management

- **Host pointers** point to CPU memory and **cannot** be dereferenced on the GPU.
- **Device pointers** point to GPU memory and **cannot** be dereferenced on the CPU.
- Key functions (similar to malloc/free/memcpy):  
  - **cudaMalloc(void **ptr, size_t size)**: allocate device memory.
  - **cudaMemcpy(dst, src, size, kind)**: copy between host/device.
  - **cudaFree(ptr)**: free device memory.

**Bottom Line:** Treat host and device memory as two separate worlds, and always use cudaMalloc / cudaMemcpy / cudaFree to move data between them.

***

### Blocks, Threads, and Indexing (1D)

- CUDA runs many copies of a kernel in parallel; each copy is a **thread**, grouped into **blocks**, blocks form a **grid**.
- Built-in variables:  
  - **blockIdx.x** = which block in the grid.
  - **threadIdx.x** = which thread inside the block.
  - **blockDim.x** = number of threads per block.
- A unique 1D index per thread (for vector/array) is:  
  - `int index = threadIdx.x + blockIdx.x * blockDim.x;`.

**Bottom Line:** Each thread needs a unique index so it knows “which element is mine”; combine block and thread indices to get that index.

***

### Vector Addition Examples

- **One block, many threads:**  
  - Kernel: `c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];`.
  - Launch: `add<<<1,N>>>(d_a, d_b, d_c);`.
- **Many blocks, one thread each:**  
  - Kernel: `c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];`.
  - Launch: `add<<<N,1>>>(d_a, d_b, d_c);`.
- **Blocks + threads:**  
  - Kernel uses `index = threadIdx.x + blockIdx.x * blockDim.x;` then `c[index] = a[index] + b[index];`.
  - Launch: `add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(...);`.

**Bottom Line:** Vector add is the “hello world” of CUDA: every thread just adds one pair of elements and writes one result.

***

### Handling Arbitrary N (Bounds Check)

- Real **N** is often not a clean multiple of **blockDim.x**.
- Launch trick: `add<<<(N + M - 1)/M, M>>>(...);` for M threads per block.
- Inside kernel: check `if (index < n)` before accessing arrays.

**Bottom Line:** Always guard memory access with `if (index < n)` when your launch may produce extra threads.

***

### Why Threads (Not Only Blocks)

- Blocks are **independent**; they cannot directly talk to each other.
- Threads inside the **same block** can:  
  - Communicate via **shared memory**.
  - Synchronize with **__syncthreads()**.

**Bottom Line:** Threads inside a block are like teammates in one room who can share a whiteboard (shared memory), while blocks are like separate rooms that cannot talk.

***

### Shared Memory & __syncthreads() with 1D Stencil

- **Shared memory** (declared with **__shared__**) is fast on-chip memory visible to threads in the same block only.
- In the 1D stencil:  
  - Each thread handles one output element.
  - All threads copy needed input into **shared** array `temp`.
  - Extra elements at edges are called **halo**.
- **__syncthreads()** is a barrier: all threads in the block wait until everyone reaches that line.

**Bottom Line:** Shared memory + __syncthreads() turn many threads into a coordinated team that first loads data together, then safely computes.

***

### Data Race and Fix in Stencil

- Without sync, a thread may read **temp[...]** before another thread has written its halo values.
- This is a **data race**: order of reads/writes is unpredictable.
- Solution: after loading shared memory, call **__syncthreads()** before using `temp`.

**Bottom Line:** Always synchronize threads (with __syncthreads) after shared loading and before using shared data, or results will be random and buggy.

***

### Managing Device Execution & Errors

- **Kernel launches are asynchronous**: CPU continues immediately; GPU runs in background.
- Ways to synchronize CPU with GPU:  
  - **cudaMemcpy()**: blocks until copy + previous CUDA calls are done.
  - **cudaMemcpyAsync()**: non-blocking; combine with **cudaDeviceSynchronize()** if you need to wait.
- Error handling functions:  
  - **cudaGetLastError()** → last error.
  - **cudaGetErrorString(err)** → printable description.

**Bottom Line:** The GPU runs independently; use memcpy/sync calls and error-checking to be sure work really finished and succeeded.

***

### Device Selection & Compute Capability

- You can query and choose GPUs:  
  - **cudaGetDeviceCount**, **cudaSetDevice**, **cudaGetDevice**, **cudaGetDeviceProperties**.
- **Compute capability** describes GPU architecture features (registers, memory sizes, double precision, caches, etc.).
- Different compute capabilities (1.x, 2.x, etc.) support different features like double precision or concurrent kernels.

**Bottom Line:** Not all GPUs are equal; use device queries and compute capability to know what your code can safely use.

***

### Multi-Dimensional IDs and Textures (Overview)

- **threadIdx**, **blockIdx**, **blockDim**, **gridDim** can be 1D, 2D, or 3D: `.x`, `.y`, `.z`.
- This helps map naturally to 2D/3D data like images and volumes.
- **Textures**:  
  - Read-only objects with special **caches** and **filtering** (linear, bilinear, trilinear).
  - Handle out-of-bounds via wrap/clamp, and can be 1D/2D/3D.

**Bottom Line:** CUDA’s multi-dimensional indices and textures make it easier and faster to handle images and 3D data.

***

### CUDA Summary (From Slides)

- To write CUDA programs you must know:  
  - **__global__**, **blockIdx.x**, **threadIdx.x**, `<<< >>>`.
  - **cudaMalloc**, **cudaMemcpy**, **cudaFree** for memory.
  - **__shared__**, **__syncthreads**, **cudaMemcpy vs cudaMemcpyAsync**, **cudaDeviceSynchronize**.

**Bottom Line:** CUDA basics = how to launch kernels, index threads, manage GPU memory, and synchronize/communicate efficiently.

***

### Matrix Multiplication on CPU vs GPU

- Goal: given **two N×N matrices** X and Y, compute result matrix **ans** = X × Y.
- Math rule: each output **C[i][j] = sum over k of A[i][k] * B[k][j]**.
- 2D → 1D index formula (row-major):  
  - `index(row, col) = row * cols + col`.
  - So `A[i][k] = A[i*CA + k]`, `B[k][j] = B[k*CB + j]`, `C[i][j] = C[i*CB + j]`.

**Bottom Line:** Matrix multiply builds each cell from a row and a column, and on GPU you usually store and index matrices as simple 1D arrays.

***

### Why 1D Arrays Matter

- Even 2D matrices are stored as one long array in memory like `[a00, a01, a02, a10, a11, a12]`.
- The 1D index lets CUDA threads work easily since each thread often uses one linear index `idx`.

**Bottom Line:** GPUs love 1D arrays; you convert (row, col) to a single index so each thread can work with a contiguous memory model.

***

### CUDA Execution Model for Matrix Multiply

- A **kernel launch** creates many threads; each thread runs the **same** kernel code but on **different data**.
- Simplest mapping for N×N:  
  - **N*N threads**, each computing exactly **one** output cell.
- Each thread has a unique linear id **idx**.

**Bottom Line:** Matrix multiply on GPU is “one thread = one output cell,” so all cells are computed in parallel.

***

### Mapping 1D idx → (row, col)

- Compute `idx = blockIdx.x * blockDim.x + threadIdx.x;`.
- Convert to coordinates:  
  - `row = idx / N;` (integer division).
  - `col = idx % N;`.
- Then compute that cell:  
  - `C[idx] = sum_k A[row*N + k] * B[k*N + col];`.

**Bottom Line:** Use division and modulo to turn a flat thread index into row and column positions in the matrix.

***

### Matrix Multiply Kernel (1D Style)

- Example kernel:  
  - Compute `idx`, `total = N*N`, and check `if (idx < total)`.
  - Derive `row` and `col` from `idx`.
  - Loop `k` from 0 to N-1, accumulate `sum += A[row*N + k] * B[k*N + col];`.
  - Store `C[idx] = sum;`.
- Each thread only reads its needed row of A and column of B and writes a single C element.

**Bottom Line:** Inside the kernel, each thread just runs a tiny for-loop over k to compute its assigned C cell.

***

### Example of What One Thread Touches

- For N=4 and `idx=6` → `(row=1, col=2)`.
- It reads: `A[4], A[5], A[6], A[7]` and `B[2], B[6], B[10], B[14]`, then writes `C[6]`.
- So one thread touches a small slice of A and B plus one output spot.

**Bottom Line:** Each thread’s footprint is small and independent, which is why GPU can run many threads safely in parallel.

***

### Boundary Check and Launch Configuration

- Need `if (idx < N*N)` because often you launch **more threads than cells** (due to block size).
- Pick:  
  - `blockSize = 256;` (common).
  - `total = N*N;`.
  - `gridSize = (total + blockSize - 1) / blockSize;`.
- Total threads = `gridSize * blockSize`, which covers at least `N*N`.

**Bottom Line:** Use grid/block math to cover the full matrix, then protect against extra threads with a simple if-condition.

***

### CPU vs GPU View

- **CPU version:** one worker does C, then C, then C, all the way to C[N*N-1].
- **GPU version:** many workers; each thread computes **one C[idx]** at the same time.

**Bottom Line:** GPU turns a long serial matrix multiply loop into a huge set of independent small tasks done in parallel.

***

## Supporting Details (Likely to Be Tested)

- **Hello World CUDA:** you can compile plain C code with `nvcc` even without device code, and then extend it by adding a trivial kernel and launch.
- The add-on device example allocates device memory, copies inputs, launches `add<<<1,1>>>`, copies result back, and frees memory.
- **Reviews** in slides highlight:  
  - Host vs device roles.
  - Use of CUDA keywords and APIs (`__global__`, `blockIdx.x`, `threadIdx.x`, `cudaMalloc`, `cudaMemcpy`, `cudaFree`).
  - Use of shared memory and __syncthreads for safe cooperation.

**Bottom Line:** Exam questions will likely poke your memory of the standard CUDA pattern: allocate–copy–launch–sync–copy back–free.

***

## Definitions (Memorize-Friendly)

- **Host:** CPU and its memory running normal C/C++ code.
- **Device:** GPU and its memory running CUDA kernels.
- **Kernel:** A function marked with `__global__` that runs on the GPU and is launched from the CPU.
- **Block:** A group of threads that can share **shared memory** and use **__syncthreads()**.
- **Grid:** Collection of blocks launched by one kernel call.
- **threadIdx, blockIdx, blockDim, gridDim:** Built-in variables that give thread/block positions and dimensions (can be 1D/2D/3D).
- **Shared memory:** Fast on-chip memory shared only by threads in the same block, declared with `__shared__`.
- **__syncthreads():** Barrier that makes all threads in a block wait until everyone reaches that line.
- **Data race:** Bug where threads read/write the same data without proper order or sync, leading to unpredictable results.
- **cudaMalloc / cudaFree / cudaMemcpy:** CUDA API calls for allocating, freeing, and copying device memory.
- **Compute capability:** Version number describing GPU architecture features (memory sizes, instructions, etc.).

**Bottom Line:** Knowing these short definitions lets you quickly parse exam questions and code snippets.
