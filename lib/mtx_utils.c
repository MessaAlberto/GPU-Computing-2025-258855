#include "../include/mtx_utils.h"

void COO_to_CSR(int nrows, int ncols, int nnz,
                const int* row_coo, const int* col_coo,
                const double* values_coo,
                CSR_matrix* matrix) {
  matrix->nrows = nrows;
  matrix->ncols = ncols;
  matrix->nnz = nnz;

  matrix->row_ptr = (int*)malloc((nrows + 1) * sizeof(int));
  matrix->col_idx = (int*)malloc(nnz * sizeof(int));
  matrix->values = (double*)malloc(nnz * sizeof(double));

  if (!matrix->row_ptr || !matrix->col_idx || !matrix->values) {
    fprintf(stderr, "CSR memory allocation failed\n");
    free(matrix->row_ptr);
    free(matrix->col_idx);
    free(matrix->values);
    exit(EXIT_FAILURE);
  }

  // Initialize the row pointer
  memset(matrix->row_ptr, 0, (nrows + 1) * sizeof(int));
  for (int i = 0; i < nnz; i++) {
    matrix->row_ptr[row_coo[i] + 1]++;
  }
  for (int i = 1; i <= nrows; i++) {
    matrix->row_ptr[i] += matrix->row_ptr[i - 1];
  }

  int* row_counts = (int*)calloc(nrows, sizeof(int));
  if (!row_counts) {
    fprintf(stderr, "Temporary rows memory allocation failed\n");
    free(matrix->row_ptr);
    free(matrix->col_idx);
    free(matrix->values);
    free(row_counts);
    exit(EXIT_FAILURE);
  }

  // Sort the COO data by row and column indices
  for (int i = 0; i < nnz; i++) {
    int row = row_coo[i];
    int idx = matrix->row_ptr[row] + row_counts[row]++;
    matrix->col_idx[idx] = col_coo[i];
    matrix->values[idx] = values_coo[i];
  }

  for (int i = 0; i < nrows; i++) {
    int start = matrix->row_ptr[i];
    int end = matrix->row_ptr[i + 1];
    int row_size = end - start;

    // Sort in same row by column index
    for (int j = start + 1; j < end; j++) {
      int key_col = matrix->col_idx[j];
      double key_val = matrix->values[j];
      int k = j - 1;

      while (k >= start && matrix->col_idx[k] > key_col) {
        matrix->col_idx[k + 1] = matrix->col_idx[k];
        matrix->values[k + 1] = matrix->values[k];
        k--;
      }

      matrix->col_idx[k + 1] = key_col;
      matrix->values[k + 1] = key_val;
    }
  }

  free(row_counts);
}

void read_CSR_mtx(const char* filename, CSR_matrix* matrix) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    exit(EXIT_FAILURE);
  }

  char line[1024];
  do {
    if (!fgets(line, sizeof(line), file)) {
      fprintf(stderr, "Error reading file: %s\n", filename);
      fclose(file);
      exit(EXIT_FAILURE);
    }
  } while (line[0] == '%');

  // Read the matrix dimensions and number of non-zero elements
  sscanf(line, "%d %d %d", &matrix->nrows, &matrix->ncols, &matrix->nnz);

  // Matrix file is in COO format
  int* row_coo = (int*)malloc(matrix->nnz * sizeof(int));
  int* col_coo = (int*)malloc(matrix->nnz * sizeof(int));
  double* values_coo = (double*)malloc(matrix->nnz * sizeof(double));

  if (!row_coo || !col_coo || !values_coo) {
    fprintf(stderr, "COO memory allocation failed\n");
    fclose(file);
    free(row_coo);
    free(col_coo);
    free(values_coo);
    exit(EXIT_FAILURE);
  }

  // Read the COO data
  for (int i = 0; i < matrix->nnz; i++) {
    int row, col;
    double val;
    if (fscanf(file, "%d %d %lf", &row, &col, &val) != 3) {
      fprintf(stderr, "Error reading matrix data from file: %s\n", filename);
      free(row_coo);
      free(col_coo);
      free(values_coo);
      fclose(file);
      exit(EXIT_FAILURE);
    }
    row_coo[i] = row - 1;
    col_coo[i] = col - 1;
    values_coo[i] = val;
  }

  // Convert COO to CSR format
  COO_to_CSR(matrix->nrows, matrix->ncols, matrix->nnz,
             row_coo, col_coo, values_coo, matrix);

  fclose(file); 
  free(row_coo);
  free(col_coo);
  free(values_coo);
}

double* init_RandVector(int n) {
  double* vec = (double*)malloc(n * sizeof(double));
  if (!vec) {
    fprintf(stderr, "Vector memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  srand(52);

  for (int i = 0; i < n; i++) {
    double val = ((double)rand() / RAND_MAX) * 200.0 - 100.0; // random value between -100 and 100
    double scale = pow(10.0, rand() % 7 - 3); // scale between 0.001 and 1000
    vec[i] = val * scale;
  }

  return vec;
}