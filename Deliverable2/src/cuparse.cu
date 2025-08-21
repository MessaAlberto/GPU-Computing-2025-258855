#include "../include/mtx_utils.h"
#include "../include/test_utils.h"
#include <cusparse.h>

// ===================== Helpers & safety =====================
#define CUDA_CHECK(stmt)                                                 \
  do {                                                                   \
    cudaError_t _err = (stmt);                                           \
    if (_err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,      \
              cudaGetErrorString(_err));                                 \
      return 1;                                                          \
    }                                                                    \
  } while (0)

#define CUSPARSE_CHECK(stmt)                                             \
  do {                                                                   \
    cusparseStatus_t _err = (stmt);                                      \
    if (_err != CUSPARSE_STATUS_SUCCESS) {                               \
      fprintf(stderr, "cuSPARSE Error %s:%d: %d\n", __FILE__, __LINE__,  \
              _err);                                                     \
      return 1;                                                          \
    }                                                                    \
  } while (0)

// ===================== Main =====================

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <matrix_file> <block_dim>\n", argv[0]);
    return -1;
  }

  const int BLOCK_DIM = atoi(argv[2]);
  if (BLOCK_DIM <= 0 || BLOCK_DIM > 1024) {
    fprintf(stderr, "Error: Invalid block dimension. Must be between 1 and 1024.\n");
    return 1;
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

  // Read the matrix from file
  read_COO_mtx(argv[1], &coo_matrix);

  // Allocate memory for the CSR matrix and vectors
  matrix.nrows = coo_matrix.nrows;
  matrix.ncols = coo_matrix.ncols;
  matrix.nnz   = coo_matrix.nnz;
  CUDA_CHECK(cudaMallocManaged(&matrix.row_ptr, (matrix.nrows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&matrix.col_idx, matrix.nnz * sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&matrix.values, matrix.nnz * sizeof(double)));
  CUDA_CHECK(cudaMallocManaged(&vec,            matrix.ncols * sizeof(double)));
  CUDA_CHECK(cudaMallocManaged(&result,         matrix.nrows * sizeof(double)));

  // Convert COO to CSR format and initialize the vector
  COO_to_CSR(&coo_matrix, &matrix);
  free(coo_matrix.row_idx);
  free(coo_matrix.col_idx);
  free(coo_matrix.values);
  init_RandVector(vec, matrix.ncols);

  // Prefetch data to the GPU
  CUDA_CHECK(cudaMemPrefetchAsync(matrix.row_ptr, (matrix.nrows + 1) * sizeof(int), device));
  CUDA_CHECK(cudaMemPrefetchAsync(matrix.col_idx, matrix.nnz * sizeof(int), device));
  CUDA_CHECK(cudaMemPrefetchAsync(matrix.values, matrix.nnz * sizeof(double), device));
  CUDA_CHECK(cudaMemPrefetchAsync(vec,            matrix.ncols * sizeof(double), device));
  CUDA_CHECK(cudaMemPrefetchAsync(result,         matrix.nrows * sizeof(double), device));
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- cuSPARSE setup ---
  cusparseHandle_t handle;
  CUSPARSE_CHECK(cusparseCreate(&handle));

  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void* dBuffer = NULL;
  size_t bufferSize = 0;

  // Create sparse matrix A in CSR format
  CUSPARSE_CHECK(cusparseCreateCsr(&matA, matrix.nrows, matrix.ncols, matrix.nnz,
                                   matrix.row_ptr, matrix.col_idx, matrix.values,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

  // Create dense vectors x and y
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, matrix.ncols, vec, CUDA_R_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, matrix.nrows, result, CUDA_R_64F));

  const double alpha = 1.0;
  const double beta  = 0.0;

  // Query buffer size
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha, matA, vecX, &beta, vecY,
                                         CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                         &bufferSize));
  CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

  printf("Using kernel: \t\tcuSPARSE SpMV\n");
  printf("Using matrix: \t\t%s\n", argv[1]);
  printf("Using block size: \t%d\n\n", BLOCK_DIM);
  print_mtx_stats(&matrix);
  fflush(stdout);

  // Warm up
  for (int i = 0; i < WARM_UP; ++i) {
    CUSPARSE_CHECK(cusparseSpMV(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                dBuffer));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Benchmarking phase
  for (int i = 0; i < REP; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    CUSPARSE_CHECK(cusparseSpMV(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                dBuffer));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    times[i] = ms * 1e-3f; // Convert to seconds
  }

  // Calculate results
  const float meanTime = arithmetic_mean(times, REP); // sec
  const float flopCount = 2.0f * (float)matrix.nnz;
  const float gflops = calculate_GFlops(flopCount, meanTime);

  // Bandwidth calculation (similar to your custom kernels)
  const int Bd = sizeof(double); // 8 bytes
  const int Bi = sizeof(int);    // 4 bytes
  size_t readBytes   = (size_t)matrix.nrows * (Bi + Bi) 
                     + (size_t)matrix.nnz * (Bi + Bd + Bi);
  size_t writeBytes  = (size_t)matrix.nrows * Bd;
  const size_t totalBytes = readBytes + writeBytes;
  const float bandwidthGBs = (float)totalBytes / (meanTime * 1e9f);

  // Result sum
  CUDA_CHECK(cudaMemPrefetchAsync(result, matrix.nrows * sizeof(double), cudaCpuDeviceId));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int r = 0; r < matrix.nrows; ++r) finalResult += result[r];

  printf("Sum of resulting vector: %f\n", finalResult);
  printf("Mean time: %f ms\n", meanTime * 1e3f);
  printf("GFlops: %f\n", gflops);
  printf("Bandwidth: %f GB/s\n", bandwidthGBs);

  // --- Cleanup
  CUSPARSE_CHECK(cusparseDestroySpMat(matA));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
  CUSPARSE_CHECK(cusparseDestroy(handle));
  CUDA_CHECK(cudaFree(dBuffer));
  CUDA_CHECK(cudaFree(matrix.row_ptr));
  CUDA_CHECK(cudaFree(matrix.col_idx));
  CUDA_CHECK(cudaFree(matrix.values));
  CUDA_CHECK(cudaFree(vec));
  CUDA_CHECK(cudaFree(result));
  CUDA_CHECK(cudaDeviceReset());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
