#include "../include/mtx_utils.h"
#include "../include/test_utils.h"

#define KERNEL SpMV_Hybrid
#define KERNEL_NO_BOUNDS SpMV_Hybrid_noBounds
#define KERNEL_NAME "SpMV_Hybrid"
#define KERNEL_PARAMS                                                                  \
  matrix.row_ptr, matrix.col_idx, matrix.values, vec, result, short_rows, medium_rows, \
      long_rows, num_short, num_medium, num_long, short_blocks, medium_blocks, long_blocks

// ===================== Helpers & safety =====================
#define CUDA_CHECK(stmt)                                            \
  do {                                                              \
    cudaError_t _err = (stmt);                                      \
    if (_err != cudaSuccess) {                                      \
      fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_err));                            \
      return 1;                                                     \
    }                                                               \
  } while (0)

#define CUDA_CHECK_KERNEL()                                           \
  do {                                                                \
    cudaError_t err = cudaGetLastError();                             \
    if (err != cudaSuccess) {                                         \
      fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err)); \
      return 1;                                                       \
    }                                                                 \
  } while (0)

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

// ===================== Kernel =====================
__device__ void SpMV_Hybrid_body(
    const int *__restrict__ row_ptr, const int *__restrict__ col_idx,
    const double *__restrict__ values, const double *__restrict__ vec,
    double *__restrict__ result, const int *__restrict__ short_rows,
    const int *__restrict__ medium_rows, const int *__restrict__ long_rows, int num_short,
    int num_medium, int num_long, int short_blocks, int medium_blocks, int long_blocks) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);  // lane in warp
  const int warp_id = threadIdx.x >> 5;            // warp ID in block
  const int block_id = blockIdx.x;
  extern __shared__ double shared_mem[];  // shared memory for long rows

  // Short rows
  if (block_id < short_blocks) {
    int idx = block_id * blockDim.x + threadIdx.x;
    if (idx < num_short) {
      int r = short_rows[idx];
      double sum = 0.0;
      for (int j = row_ptr[r]; j < row_ptr[r + 1]; j++)
        sum += values[j] * __ldg(&vec[col_idx[j]]);
      result[r] = sum;
    }
    return;
  }

  // Medium rows
  if (num_medium > 0 && block_id < short_blocks + medium_blocks) {
    int mblock = block_id - short_blocks;
    int row_idx = mblock * (blockDim.x / WARP_SIZE) + warp_id;
    if (row_idx < num_medium) {
      int r = medium_rows[row_idx];
      double sum = 0.0;
      for (int j = row_ptr[r] + lane; j < row_ptr[r + 1]; j += WARP_SIZE)
        sum += values[j] * __ldg(&vec[col_idx[j]]);
      for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
      if (lane == 0) result[r] = sum;
    }
    return;
  }

  // Long rows with shared memory multi-warp reduction
  int lblock = block_id - short_blocks - medium_blocks;
  if (lblock >= num_long) return;

  int r = long_rows[lblock];
  double thread_sum = 0.0;

  // Each thread computes its partial sum
  for (int j = row_ptr[r] + threadIdx.x; j < row_ptr[r + 1]; j += blockDim.x)
    thread_sum += values[j] * __ldg(&vec[col_idx[j]]);

  // Intra-warp reduction
  for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
    thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, off);

  int warp_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_idx = threadIdx.x >> 5;

  // Lane 0 of each warp writes its sum to shared memory
  if (warp_lane == 0) shared_mem[warp_idx] = thread_sum;
  __syncthreads();

  // First warp reduces all warp sums
  if (warp_idx == 0) {
    double final_sum = (warp_lane < (blockDim.x / WARP_SIZE)) ? shared_mem[warp_lane] : 0.0;
    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
      final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, off);
    if (warp_lane == 0) result[r] = final_sum;
  }
}

__global__ void SpMV_Hybrid_noBounds(
    const int *__restrict__ row_ptr, const int *__restrict__ col_idx,
    const double *__restrict__ values, const double *__restrict__ vec,
    double *__restrict__ result, const int *__restrict__ short_rows,
    const int *__restrict__ medium_rows, const int *__restrict__ long_rows, int num_short,
    int num_medium, int num_long, int short_blocks, int medium_blocks, int long_blocks) {
  SpMV_Hybrid_body(row_ptr, col_idx, values, vec, result, short_rows, medium_rows, long_rows,
                   num_short, num_medium, num_long, short_blocks, medium_blocks, long_blocks);
}

template <int minBlocksPerSM>
__global__ __launch_bounds__(BLOCK_DIM, minBlocksPerSM) void SpMV_Hybrid(
    const int *__restrict__ row_ptr, const int *__restrict__ col_idx,
    const double *__restrict__ values, const double *__restrict__ vec,
    double *__restrict__ result, const int *__restrict__ short_rows,
    const int *__restrict__ medium_rows, const int *__restrict__ long_rows, int num_short,
    int num_medium, int num_long, int short_blocks, int medium_blocks, int long_blocks) {
  SpMV_Hybrid_body(row_ptr, col_idx, values, vec, result, short_rows, medium_rows, long_rows,
                   num_short, num_medium, num_long, short_blocks, medium_blocks, long_blocks);
}

// ===================== Main =====================
int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <matrix_file>\n", argv[0]);
    return -1;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  double finalResult = 0.0;
  float times[REP] = {0.0f};
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Matrix, vector, and result vector
  COO_matrix coo_matrix{};
  CSR_matrix matrix{};
  double *vec = nullptr;
  double *result = nullptr;

  // Matrix statistics
  int max_nnz = 0;
  int min_nnz = INT_MAX;
  double avg_nnzPerRow = 0.0;

  // Read the matrix from file
  read_COO_mtx(argv[1], &coo_matrix);

  // Allocate memory for the CSR matrix and vectors
  matrix.nrows = coo_matrix.nrows;
  matrix.ncols = coo_matrix.ncols;
  matrix.nnz = coo_matrix.nnz;
  CUDA_CHECK(cudaMallocManaged(&matrix.row_ptr, (matrix.nrows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&matrix.col_idx, matrix.nnz * sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&matrix.values, matrix.nnz * sizeof(double)));
  CUDA_CHECK(cudaMallocManaged(&vec, matrix.ncols * sizeof(double)));
  CUDA_CHECK(cudaMallocManaged(&result, matrix.nrows * sizeof(double)));

  // Convert COO to CSR format and initialize the vector
  COO_to_CSR(&coo_matrix, &matrix);
  free(coo_matrix.row_idx);
  free(coo_matrix.col_idx);
  free(coo_matrix.values);
  init_RandVector(vec, matrix.ncols);

  // Advise + Prefetch (safe: size>0 e pointer managed)
  if (matrix.nnz > 0) {
    CUDA_CHECK(cudaMemAdvise(matrix.values, matrix.nnz * sizeof(double),
                             cudaMemAdviseSetPreferredLocation, device));
    CUDA_CHECK(cudaMemAdvise(matrix.col_idx, matrix.nnz * sizeof(int),
                             cudaMemAdviseSetPreferredLocation, device));
  }
  CUDA_CHECK(cudaMemAdvise(matrix.row_ptr, (matrix.nrows + 1) * sizeof(int),
                           cudaMemAdviseSetReadMostly, device));
  CUDA_CHECK(
      cudaMemAdvise(vec, matrix.ncols * sizeof(double), cudaMemAdviseSetReadMostly, device));

  CUDA_CHECK(cudaMemPrefetchAsync(matrix.row_ptr, (matrix.nrows + 1) * sizeof(int), device));
  if (matrix.nnz > 0) {
    CUDA_CHECK(cudaMemPrefetchAsync(matrix.col_idx, matrix.nnz * sizeof(int), device));
    CUDA_CHECK(cudaMemPrefetchAsync(matrix.values, matrix.nnz * sizeof(double), device));
  }
  CUDA_CHECK(cudaMemPrefetchAsync(vec, matrix.ncols * sizeof(double), device));
  CUDA_CHECK(cudaMemPrefetchAsync(result, matrix.nrows * sizeof(double), device));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Classify rows
  int *host_short = (int *)malloc(matrix.nrows * sizeof(int));
  int *host_medium = (int *)malloc(matrix.nrows * sizeof(int));
  int *host_long = (int *)malloc(matrix.nrows * sizeof(int));
  int num_short = 0, num_medium = 0, num_long = 0;

  const int short_threshold = WARP_SIZE * 2;
  const int medium_threshold = WARP_SIZE * 16;

  classify_rows(matrix.row_ptr, matrix.nrows, host_short, host_medium, host_long, &num_short,
                &num_medium, &num_long, short_threshold, medium_threshold);

  int *short_rows = nullptr, *medium_rows = nullptr, *long_rows = nullptr;
  if (num_short > 0) {
    CUDA_CHECK(cudaMallocManaged(&short_rows, num_short * sizeof(int)));
    memcpy(short_rows, host_short, num_short * sizeof(int));
  }
  if (num_medium > 0) {
    CUDA_CHECK(cudaMallocManaged(&medium_rows, num_medium * sizeof(int)));
    memcpy(medium_rows, host_medium, num_medium * sizeof(int));
  }
  if (num_long > 0) {
    CUDA_CHECK(cudaMallocManaged(&long_rows, num_long * sizeof(int)));
    memcpy(long_rows, host_long, num_long * sizeof(int));
  }

  free(host_short);
  free(host_medium);
  free(host_long);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Grid configuration
  int short_blocks = (num_short + BLOCK_DIM - 1) / BLOCK_DIM;
  int medium_blocks = (num_medium + (BLOCK_DIM / WARP_SIZE) - 1) / (BLOCK_DIM / WARP_SIZE);
  int long_blocks = num_long;
  int total_blocks = short_blocks + medium_blocks + long_blocks;

  printf("Using kernel: \t\t%s\n", KERNEL_NAME);
  printf("Using matrix: \t\t%s\n", argv[1]);
  printf("Using block size: \t%d\n\n", BLOCK_DIM);
  print_mtx_stats(&matrix, &max_nnz, &min_nnz, &avg_nnzPerRow);
  fflush(stdout);

  int minBlocksPerSM = suggest_minBlocksPerSM(&matrix, BLOCK_DIM, max_nnz, avg_nnzPerRow);

  // ***** SHARED MEMORY SIZE *****
  // Deve contenere: CHUNK_SIZE + num_warps elementi double
  const int CHUNK_SIZE = 128;
  const int num_warps = BLOCK_DIM / WARP_SIZE;
  const size_t SHMEM_BYTES = (size_t)(CHUNK_SIZE + num_warps) * sizeof(double);

  // Warm up
  for (int i = 0; i < WARM_UP; ++i) {
    switch (minBlocksPerSM) {
      case 0:
        KERNEL_NO_BOUNDS<<<total_blocks, BLOCK_DIM, SHMEM_BYTES>>>(KERNEL_PARAMS);
        break;
      case 1:
        KERNEL<1><<<total_blocks, BLOCK_DIM, SHMEM_BYTES>>>(KERNEL_PARAMS);
        break;
      case 2:
        KERNEL<2><<<total_blocks, BLOCK_DIM, SHMEM_BYTES>>>(KERNEL_PARAMS);
        break;
      case 3:
        KERNEL<3><<<total_blocks, BLOCK_DIM, SHMEM_BYTES>>>(KERNEL_PARAMS);
        break;
      default:
        KERNEL_NO_BOUNDS<<<total_blocks, BLOCK_DIM, SHMEM_BYTES>>>(KERNEL_PARAMS);
        break;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(err));
      cudaDeviceReset();
      return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_KERNEL();
  }

  // Benchmarking phase
  for (int i = 0; i < REP; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    switch (minBlocksPerSM) {
      case 0:
        KERNEL_NO_BOUNDS<<<total_blocks, BLOCK_DIM, SHMEM_BYTES>>>(KERNEL_PARAMS);
        break;
      case 1:
        KERNEL<1><<<total_blocks, BLOCK_DIM, SHMEM_BYTES>>>(KERNEL_PARAMS);
        break;
      case 2:
        KERNEL<2><<<total_blocks, BLOCK_DIM, SHMEM_BYTES>>>(KERNEL_PARAMS);
        break;
      case 3:
        KERNEL<3><<<total_blocks, BLOCK_DIM, SHMEM_BYTES>>>(KERNEL_PARAMS);
        break;
      default:
        KERNEL_NO_BOUNDS<<<total_blocks, BLOCK_DIM, SHMEM_BYTES>>>(KERNEL_PARAMS);
        break;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(err));
      cudaDeviceReset();
      return 1;
    }
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK_KERNEL();

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    times[i] = ms * 1e-3f;
  }

  // Calculate results
  const float meanTime = arithmetic_mean(times, REP);  // sec
  const float flopCount = 2.0f * (float)matrix.nnz;
  const float gflops = calculate_GFlops(flopCount, meanTime);

  // Bandwidth calc (worst-case)
  const int Bd = sizeof(double);
  const int Bi = sizeof(int);
  size_t readBytes = (size_t)matrix.nrows * (Bi + Bi) + (size_t)matrix.nnz * (Bi + Bd + Bi);
  size_t writeBytes = (size_t)matrix.nrows * Bd;
  const size_t totalBytes = readBytes + writeBytes;
  const float bandwidthGBs = (float)totalBytes / (meanTime * 1e9f);

  // Copy result (prefetch to CPU ok per UVM)
  CUDA_CHECK(cudaMemPrefetchAsync(result, matrix.nrows * sizeof(double), cudaCpuDeviceId));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int r = 0; r < matrix.nrows; ++r) finalResult += result[r];

  printf("Sum of resulting vector: %f\n", finalResult);
  printf("Mean time: %f ms\n", meanTime * 1e3f);
  printf("GFlops: %f\n", gflops);
  printf("Bandwidth: %f GB/s\n", bandwidthGBs);

  if (num_short > 0) CUDA_CHECK(cudaFree(short_rows));
  if (num_medium > 0) CUDA_CHECK(cudaFree(medium_rows));
  if (num_long > 0) CUDA_CHECK(cudaFree(long_rows));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(matrix.row_ptr));
  CUDA_CHECK(cudaFree(matrix.col_idx));
  CUDA_CHECK(cudaFree(matrix.values));
  CUDA_CHECK(cudaFree(vec));
  CUDA_CHECK(cudaFree(result));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
