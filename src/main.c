#include "../include/mtx_utils.h"
#include "../include/test_utils.h"

void do_SpMV(const CSR_matrix* matrix, const double* vec, double* result) {
  for (int i = 0; i < matrix->nrows; i++) {
    result[i] = 0.0;
    for (int j = matrix->row_ptr[i]; j < matrix->row_ptr[i + 1]; j++) {
      result[i] += matrix->values[j] * vec[matrix->col_idx[j]];
    }
  }
}

void do_SpMV_cache_optimised(const CSR_matrix* matrix, const double* vec, double* result) {
  int* row_ptr = matrix->row_ptr;
  int* col_idx = matrix->col_idx;
  double* values = matrix->values;
  int nrows = matrix->nrows;

  for (int i = 0; i < nrows; i++) {
    double sum = 0.0;
    int row_start = row_ptr[i];
    int row_end = row_ptr[i + 1];

    // Unrolling the inner loop by 4
    int j = row_ptr[i];
    int end = row_ptr[i + 1];

    for (; j + 3 < end; j += 4) {
      sum += values[j] * vec[col_idx[j]];
      sum += values[j + 1] * vec[col_idx[j + 1]];
      sum += values[j + 2] * vec[col_idx[j + 2]];
      sum += values[j + 3] * vec[col_idx[j + 3]];
    }
    for (; j < end; j++) {
      sum += values[j] * vec[col_idx[j]];
    }
    result[i] = sum;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <matrix_file>\n", argv[0]);
    return 1;
  }

  double finalResult = 0.0;
  float times[REP] = {0};

  // Matrix, vector, and result vector
  COO_matrix coo_matrix;
  CSR_matrix matrix;
  double* vec = NULL;
  double* result = NULL;

  // Read the matrix from file
  read_COO_mtx(argv[1], &coo_matrix);

  // Allocate memory for CSR matrix and vectors
  matrix.nrows = coo_matrix.nrows;
  matrix.ncols = coo_matrix.ncols;
  matrix.nnz = coo_matrix.nnz;
  matrix.row_ptr = (int*)malloc((matrix.nrows + 1) * sizeof(int));
  matrix.col_idx = (int*)malloc(matrix.nnz * sizeof(int));
  matrix.values = (double*)malloc(matrix.nnz * sizeof(double));
  vec = (double*)malloc(matrix.ncols * sizeof(double));
  result = (double*)malloc(matrix.nrows * sizeof(double));

  if (!matrix.row_ptr || !matrix.col_idx || !matrix.values || !vec || !result) {
    fprintf(stderr, "Memory allocation for CSR matrix failed\n");
    free(coo_matrix.row_idx);
    free(coo_matrix.col_idx);
    free(coo_matrix.values);
    free(matrix.row_ptr);
    free(matrix.col_idx);
    free(matrix.values);
    free(vec);
    free(result);
    return 1;
  }

  // Convert COO to CSR format and initialize the vector
  COO_to_CSR(&coo_matrix, &matrix);
  free(coo_matrix.row_idx);
  free(coo_matrix.col_idx);
  free(coo_matrix.values);
  init_RandVector(vec, matrix.ncols);

  #ifdef USE_OPTIM
    printf("Using Optimized CPU SpMV with cache optimization\n");
  #else
    printf("Using naive CPU SpMV\n");
  #endif
  printf("Using matrix: \t\t%s\n\n", argv[1]);
  print_mtx_stats(&matrix);
  fflush(stdout);

  // Warm-up phase
  for (int i = 0; i < WARM_UP; i++) {
  #ifdef USE_OPTIM
      do_SpMV_cache_optimised(&matrix, vec, result);
  #else
      do_SpMV(&matrix, vec, result);
  #endif

    // Check the result
    if (i == 0) {
      for (int j = 0; j < matrix.nrows; j++) {
        finalResult += result[j];
      }
    }
  }

  // Benchmarking phase
  for (int i = 0; i < REP; i++) {
    TIMER_DEF(0);
    TIMER_START(0);

  #ifdef USE_OPTIM
      do_SpMV_cache_optimised(&matrix, vec, result);
  #else
      do_SpMV(&matrix, vec, result);
  #endif

    TIMER_STOP(0);
    times[i] = TIMER_ELAPSED(0) / 1.e6;  // Convert to seconds
  }

  // Print the results
  float meanTime = arithmetic_mean(times, REP);
  float flopCount = 2 * matrix.nnz;  // 2 flops per non-zero entry: multiply and add
  float gflops = calculate_GFlops(flopCount, meanTime);

  size_t readedBytes = matrix.nnz * (sizeof(int) + sizeof(double)) +  // col_idx and values
                       (matrix.nrows + 1) * sizeof(int) +             // row_ptr
                       matrix.ncols * sizeof(double);                 // vec
  size_t writtenBytes = matrix.nrows * sizeof(double);                // result

  size_t totalBytes = readedBytes + writtenBytes;
  float bandwidth = (float)totalBytes / (meanTime * 1e9);  // GB/s

  printf("Sum of resulting vector: %f\n", finalResult);
  printf("Mean time: %f ms\n", meanTime * 1e3);
  printf("GFlops: %f\n", gflops);
  printf("Bandwidth: %f GB/s\n", bandwidth);

  // Free allocated memory
  free(matrix.row_ptr);
  free(matrix.col_idx);
  free(matrix.values);
  free(vec);
  free(result);

  return 0;
}