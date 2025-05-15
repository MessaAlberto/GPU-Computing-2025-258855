#include "../include/mtx_utils.h"
#include "../include/test_utils.h"

#ifndef SELECT_KERNEL
  #error "Please define SELECT_KERNEL to select the kernel to use."
#endif

#if SELECT_KERNEL == 1
  #define KERNEL SpMV_OneThreadPerRow
  #define KERNEL_NAME "SpMV_OneThreadPerRow"
  #define KERNEL_PARAMS matrix.nrows, matrix.row_ptr, matrix.col_idx, matrix.values, vec, result
#elif SELECT_KERNEL == 2
  #define KERNEL SpMV_OneWarpPerRow
  #define KERNEL_NAME "SpMV_OneWarpPerRow"
  #define KERNEL_PARAMS matrix.nrows, matrix.row_ptr, matrix.col_idx, matrix.values, vec, result
#elif SELECT_KERNEL == 3
  #define KERNEL SpMV_coalescedBins
  #define KERNEL_NAME "SpMV_coalescedBins"
  #define KERNEL_PARAMS matrix.nrows, matrix.row_ptr, matrix.col_idx, matrix.values, vec, result, bin_rows
#endif

#define CUDA_CHECK_KERNEL()                                         \
  do {                                                              \
    cudaError_t err = cudaGetLastError();                           \
    if (err != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
      return 1;                                                     \
    }                                                               \
  } while (0)

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__global__ void SpMV_OneThreadPerRow(const int rows, const int *row_ptr, const int *col_idx,
                                     const double *values, const double *vec, double *result) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows) {
    double sum = 0.0;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    for (int j = start; j < end; j++) {
      sum += values[j] * __ldg(&vec[col_idx[j]]);
    }
    result[row] = sum;
  }
}

__global__ void SpMV_OneWarpPerRow(const int rows, const int *row_ptr, const int *col_idx,
                                   const double *values, const double *vec, double *result) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  if (warp_id < rows) {
    double sum = 0.0;
    int start = row_ptr[warp_id];
    int end = row_ptr[warp_id + 1];
    for (int j = start + lane_id; j < end; j += WARP_SIZE) {
      sum += values[j] * __ldg(&vec[col_idx[j]]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (lane_id == 0) {
      result[warp_id] = sum;
    }
  }
}

__global__ void SpMV_coalescedBins(const int num_bins, const int *row_ptr, const int *col_idx,
                                   const double *values, const double *vec, double *result,
                                   const int *bin_rows) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  if (warp_id >= num_bins) return;

  int row_start = bin_rows[warp_id];
  int row_end = bin_rows[warp_id + 1];

  int nnz_start = row_ptr[row_start];
  int nnz_end = row_ptr[row_end];
  int total_nnz = nnz_end - nnz_start;

  for (int i = lane_id; i < total_nnz; i += WARP_SIZE) {
    int idx = nnz_start + i;
    int row = row_start;
    while (idx >= row_ptr[row + 1] && row < row_end - 1) {
      row++;
    }

    double val = values[idx] * __ldg(&vec[col_idx[idx]]);
    atomicAdd(&result[row], val);
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <matrix_file>\n", argv[0]);
    return -1;
  }

  int device = -1;
  cudaGetDevice(&device);

  if (device == -1) {
    fprintf(stderr, "Error: No GPU device found.\n");
    return 1;
  }

  double finalResult = 0.0;
  float times[REP] = {0};
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Matrix, vector, and result vector
  COO_matrix coo_matrix;
  CSR_matrix matrix;
  double *vec = NULL;
  double *result = NULL;

  // Read the matrix from file
  read_COO_mtx(argv[1], &coo_matrix);

  // Allocate memory for the CSR matrix and vectors
  matrix.nrows = coo_matrix.nrows;
  matrix.ncols = coo_matrix.ncols;
  matrix.nnz = coo_matrix.nnz;
  cudaMallocManaged(&matrix.row_ptr, (matrix.nrows + 1) * sizeof(int));
  cudaMallocManaged(&matrix.col_idx, matrix.nnz * sizeof(int));
  cudaMallocManaged(&matrix.values, matrix.nnz * sizeof(double));
  cudaMallocManaged(&vec, matrix.ncols * sizeof(double));
  cudaMallocManaged(&result, matrix.nrows * sizeof(double));

  if (!matrix.row_ptr || !matrix.col_idx || !matrix.values || !vec || !result) {
    fprintf(stderr, "Error: Failed to allocate memory for matrix or vector.\n");
    free(coo_matrix.row_idx);
    free(coo_matrix.col_idx);
    free(coo_matrix.values);
    cudaFree(matrix.row_ptr);
    cudaFree(matrix.col_idx);
    cudaFree(matrix.values);
    cudaFree(vec);
    cudaFree(result);
    return 1;
  }

  // Convert COO to CSR format and initialize the vector
  COO_to_CSR(&coo_matrix, &matrix);
  free(coo_matrix.row_idx);
  free(coo_matrix.col_idx);
  free(coo_matrix.values);
  init_RandVector(vec, matrix.ncols);

  // Prefetch data to the GPU
  cudaMemPrefetchAsync(matrix.row_ptr, (matrix.nrows + 1) * sizeof(int), device);
  cudaMemPrefetchAsync(matrix.col_idx, matrix.nnz * sizeof(int), device);
  cudaMemPrefetchAsync(matrix.values, matrix.nnz * sizeof(double), device);
  cudaMemPrefetchAsync(vec, matrix.ncols * sizeof(double), device);
  cudaMemPrefetchAsync(result, matrix.nrows * sizeof(double), device);
  cudaDeviceSynchronize();

  // Based on the selected kernel, set the number of blocks and threads
  #if SELECT_KERNEL == 1
    int numBlocks = (matrix.nrows + BLOCK_SIZE - 1) / BLOCK_SIZE;

  #elif SELECT_KERNEL == 2
    int numBlocks = ((matrix.nrows * WARP_SIZE) + BLOCK_SIZE - 1) / BLOCK_SIZE;

  #elif SELECT_KERNEL == 3
    int *host_bin_rows = (int *)malloc((matrix.nrows + 1) * sizeof(int));
    int num_bins =
        build_coalesced_row_bins(matrix.row_ptr, matrix.nrows, host_bin_rows, WARP_SIZE);

    printf("Number of bins: %d\n", num_bins);
    for (int i = 0; i < 20; i++) {
      int start_row = host_bin_rows[i];
      int end_row = host_bin_rows[i + 1];
      int rows_in_bin = end_row - start_row;
      int nnz_in_bin = matrix.row_ptr[end_row] - matrix.row_ptr[start_row];
      printf("Bin %d: %d rows, %d NNZ\n", i, rows_in_bin, nnz_in_bin);
    }
    
    int *bin_rows = NULL;
    cudaMallocManaged(&bin_rows, (num_bins + 1) * sizeof(int));
    cudaMemcpy(bin_rows, host_bin_rows, (num_bins + 1) * sizeof(int), cudaMemcpyHostToDevice);
    free(host_bin_rows);

    cudaMemPrefetchAsync(bin_rows, (num_bins + 1) * sizeof(int), device);
    cudaDeviceSynchronize();
    
    int numBlocks = ((num_bins * WARP_SIZE) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  #endif

  printf("Using kernel: \t\t%s\n", KERNEL_NAME);
  printf("Using matrix: \t\t%s\n\n", argv[1]);
  print_mtx_stats(&matrix);

  printf("Blocks: %d, Threads per block: %d\n", numBlocks, BLOCK_SIZE);
  fflush(stdout);

  // Warm up
  for (int i = 0; i < WARM_UP; i++) {
    KERNEL<<<numBlocks, BLOCK_SIZE>>>(KERNEL_PARAMS);
    cudaDeviceSynchronize();

    CUDA_CHECK_KERNEL();

    if (i == 0) {
      for (int j = 0; j < matrix.nrows; j++) {
        finalResult += result[j];
      }
    }
  }

  // Benchmarking phase
  for (int i = 0; i < REP; i++) {
    cudaEventRecord(start);

    KERNEL<<<numBlocks, BLOCK_SIZE>>>(KERNEL_PARAMS);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CUDA_CHECK_KERNEL();

    cudaEventElapsedTime(times + i, start, stop);
  }

  // Print results
  float meanTime = arithmetic_mean(times, REP);
  float flopCount = 2.0 * matrix.nnz;
  float gflops = calculate_GFlops(flopCount, meanTime);
  printf("Final result: %f\n", finalResult);
  printf("Mean time: %f seconds\n", meanTime);
  printf("GFlops: %f\n", gflops);

  // Free memory
  cudaFree(matrix.row_ptr);
  cudaFree(matrix.col_idx);
  cudaFree(matrix.values);
  cudaFree(vec);
  cudaFree(result);
  cudaDeviceReset();
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}