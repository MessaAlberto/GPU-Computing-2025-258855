#include "../include/mtx_utils.h"
#include "../include/test_utils.h"

#define KERNEL SpMV_coalescedBins
#define KERNEL_NAME "SpMV_coalescedBins"
#define KERNEL_PARAMS \
  num_bins, matrix.row_ptr, matrix.col_idx, matrix.values, vec, result, bin_rows

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
__global__ void SpMV_coalescedBins(const int num_bins, const int *__restrict__ row_ptr,
                                   const int *__restrict__ col_idx,
                                   const double *__restrict__ values,
                                   const double *__restrict__ vec, double *__restrict__ result,
                                   const int *__restrict__ bin_rows) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int total_threads = gridDim.x * blockDim.x;
  const int warps_per_grid = total_threads / WARP_SIZE;
  int base_bw = tid / WARP_SIZE;

  for (int bw = base_bw; bw < num_bins; bw += warps_per_grid) {
    const int row_start = __ldg(&bin_rows[bw]);
    const int row_end = __ldg(&bin_rows[bw + 1]);

    for (int row = row_start; row < row_end; ++row) {
      const int start = __ldg(&row_ptr[row]);
      const int end = __ldg(&row_ptr[row + 1]);
      double sum = 0.0;

#pragma unroll
      for (int j = start + lane; j < end; j += WARP_SIZE) {
        double val = __ldg(&values[j]);
        int col = __ldg(&col_idx[j]);
        sum += val * __ldg(&vec[col]);
      }

      for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
      }

      if (lane == 0) {
        result[row] = sum;
      }
    }
  }
}

// ===================== Main =====================
int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <matrix_file>\n", argv[0]);
    return -1;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int SM = prop.multiProcessorCount;

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

  // Advise CUDA on memory usage
  CUDA_CHECK(cudaMemAdvise(matrix.values, matrix.nnz * sizeof(double),
                           cudaMemAdviseSetPreferredLocation, device));
  CUDA_CHECK(cudaMemAdvise(matrix.col_idx, matrix.nnz * sizeof(int),
                           cudaMemAdviseSetPreferredLocation, device));
  CUDA_CHECK(cudaMemAdvise(matrix.row_ptr, (matrix.nrows + 1) * sizeof(int),
                           cudaMemAdviseSetReadMostly, device));
  CUDA_CHECK(
      cudaMemAdvise(vec, matrix.ncols * sizeof(double), cudaMemAdviseSetReadMostly, device));

  // Prefetch data to the GPU
  CUDA_CHECK(cudaMemPrefetchAsync(matrix.row_ptr, (matrix.nrows + 1) * sizeof(int), device));
  CUDA_CHECK(cudaMemPrefetchAsync(matrix.col_idx, matrix.nnz * sizeof(int), device));
  CUDA_CHECK(cudaMemPrefetchAsync(matrix.values, matrix.nnz * sizeof(double), device));
  CUDA_CHECK(cudaMemPrefetchAsync(vec, matrix.ncols * sizeof(double), device));
  CUDA_CHECK(cudaMemPrefetchAsync(result, matrix.nrows * sizeof(double), device));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Set the number of blocks and threads
  int gridDim = 0;
  int *host_bin_rows = (int *)malloc((matrix.nrows + 1) * sizeof(int));
  int num_bins =
      build_coalesced_row_bins(matrix.row_ptr, matrix.nrows, host_bin_rows, WARP_SIZE);
  int *bin_rows = nullptr;
  CUDA_CHECK(cudaMallocManaged(&bin_rows, (num_bins + 1) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(bin_rows, host_bin_rows, (num_bins + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  free(host_bin_rows);
  CUDA_CHECK(cudaMemPrefetchAsync(bin_rows, (num_bins + 1) * sizeof(int), device));
  CUDA_CHECK(cudaDeviceSynchronize());

  const long long total_warps = num_bins;
  const long long total_threads = total_warps * WARP_SIZE;
  gridDim =
      (int)MIN((total_threads + BLOCK_DIM - 1) / (long long)BLOCK_DIM, (long long)SM * 8);

  printf("Using kernel: \t\t%s\n", KERNEL_NAME);
  printf("Using matrix: \t\t%s\n", argv[1]);
  printf("Using block size: \t%d\n\n", BLOCK_DIM);
  print_mtx_stats(&matrix, &max_nnz, &min_nnz, &avg_nnzPerRow);
  fflush(stdout);

  for (int i = 0; i < WARM_UP; ++i) {
    KERNEL<<<gridDim, BLOCK_DIM>>>(KERNEL_PARAMS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_KERNEL();
  }
  for (int i = 0; i < REP; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    KERNEL<<<gridDim, BLOCK_DIM>>>(KERNEL_PARAMS);
    CUDA_CHECK(cudaEventRecord(stop));
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

  // Bandwidth calculation based on the worst-case scenario
  const int Bd = sizeof(double);  // 8 bytes
  const int Bi = sizeof(int);     // 4 bytes

  size_t readBytes = (size_t)matrix.nrows * (Bi + Bi)       // row_ptr[row], row_ptr[row+1]
                     + (size_t)matrix.nnz * (Bi + Bd + Bi)  // col_idx, values, vec
                     + (size_t)(num_bins + 1) * Bi;         // bin_rows
  size_t writeBytes = (size_t)matrix.nrows * Bd;            // result[row]

  const size_t totalBytes = readBytes + writeBytes;
  const float bandwidthGBs = (float)totalBytes / (meanTime * 1e9f);

  // Copy result
  CUDA_CHECK(cudaMemPrefetchAsync(result, matrix.nrows * sizeof(double), cudaCpuDeviceId));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int r = 0; r < matrix.nrows; ++r) finalResult += result[r];

  printf("Sum of resulting vector: %f\n", finalResult);
  printf("Mean time: %f ms\n", meanTime * 1e3f);
  printf("GFlops: %f\n", gflops);
  printf("Bandwidth: %f GB/s\n", bandwidthGBs);

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
