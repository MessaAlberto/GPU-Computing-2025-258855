#include "../include/mtx_utils.h"

void do_SpMV(const CSR_matrix* matrix, const double* vec, double* result) {
  for (int i = 0; i < matrix->nrows; i++) {
    result[i] = 0.0;
    for (int j = matrix->row_ptr[i]; j < matrix->row_ptr[i + 1]; j++) {
      result[i] += matrix->values[j] * vec[matrix->col_idx[j]];
    }
  }
}

int main(int argc, char** argv) {
  CSR_matrix matrix;


  if (argc < 2) {
    fprintf(stderr, "Usage: %s <matrix_file>\n", argv[0]);
    return 1;
  }

  // Read the matrix from the file
  read_CSR_mtx(argv[1], &matrix);

  printf("Matrix dimensions: %d x %d, nnz: %d\n",
         matrix.nrows, matrix.ncols, matrix.nnz);

  double* vec = init_RandVector(matrix.ncols);
  double* result = malloc(matrix.nrows * sizeof(double));
  double finalResult = 0.0;
  if (!result) {
    fprintf(stderr, "Memory allocation for result vector failed\n");
    free(vec);
    return 1;
  }

  do_SpMV(&matrix, vec, result);

  // Check the result
  for (int i = 0; i < matrix.nrows; i++) {
    finalResult += result[i];
  }

  printf("Final result: %f\n", finalResult);

  // Free allocated memory
  free(matrix.row_ptr);
  free(matrix.col_idx);
  free(matrix.values);
  free(vec);
  free(result);

  
  return 0;
}