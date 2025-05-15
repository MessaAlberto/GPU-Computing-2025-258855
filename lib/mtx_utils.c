#include "../include/mtx_utils.h"

void read_COO_mtx(const char* filename, COO_matrix* matrix) {
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

  // Allocate memory for COO matrix
  matrix->row_idx = (int*)malloc(matrix->nnz * sizeof(int));
  matrix->col_idx = (int*)malloc(matrix->nnz * sizeof(int));
  matrix->values = (double*)malloc(matrix->nnz * sizeof(double));
  if (!matrix->row_idx || !matrix->col_idx || !matrix->values) {
    fprintf(stderr, "Memory allocation for COO matrix failed\n");
    fclose(file);
    exit(EXIT_FAILURE);
  }

  // Read the COO data
  for (int i = 0; i < matrix->nnz; i++) {
    int row, col;
    double val;
    if (fscanf(file, "%d %d %lf", &row, &col, &val) != 3) {
      fprintf(stderr, "Error reading matrix data from file: %s\n", filename);
      fclose(file);
      free(matrix->row_idx);
      free(matrix->col_idx);
      free(matrix->values);
      exit(EXIT_FAILURE);
    }
    matrix->row_idx[i] = row - 1;
    matrix->col_idx[i] = col - 1;
    matrix->values[i] = val;
  }

  fclose(file);
}

void COO_to_CSR(COO_matrix* coo_matrix, CSR_matrix* matrix) {
  // Initialize the row pointer
  memset(matrix->row_ptr, 0, (matrix->nrows + 1) * sizeof(int));
  for (int i = 0; i < matrix->nnz; i++) {
    matrix->row_ptr[coo_matrix->row_idx[i] + 1]++;
  }
  for (int i = 1; i <= matrix->nrows; i++) {
    matrix->row_ptr[i] += matrix->row_ptr[i - 1];
  }

  int* row_counts = (int*)calloc(matrix->nrows, sizeof(int));
  if (!row_counts) {
    fprintf(stderr, "Temporary rows memory allocation failed\n");
    free(coo_matrix->row_idx);
    free(coo_matrix->col_idx);
    free(coo_matrix->values);
    free(matrix->row_ptr);
    free(matrix->col_idx);
    free(matrix->values);
    free(row_counts);
    exit(EXIT_FAILURE);
  }

  // Sort the COO data by row and column indices
  for (int i = 0; i < matrix->nnz; i++) {
    int row = coo_matrix->row_idx[i];
    int idx = matrix->row_ptr[row] + row_counts[row]++;
    matrix->col_idx[idx] = coo_matrix->col_idx[i];
    matrix->values[idx] = coo_matrix->values[i];
  }

  for (int i = 0; i < matrix->nrows; i++) {
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

void init_RandVector(double* vec, int n) {
  srand(52);  // Seed for reproducibility

  for (int i = 0; i < n; i++) {
    double val = ((double)rand() / RAND_MAX) * 200.0 - 100.0; // random value between -100 and 100
    double scale = pow(10.0, rand() % 7 - 3); // scale between 0.001 and 1000
    vec[i] = val * scale;
  }
}

void print_mtx_stats(CSR_matrix* matrix) {
  printf("Matrix Statistics:\n");
  printf("  Rows: %d\n", matrix->nrows);
  printf("  Columns: %d\n", matrix->ncols);
  printf("  Non-zeros: %d\n", matrix->nnz);

  double avg_nnzPerRow = (double)matrix->nnz / matrix->nrows;
  int max_nnz = 0;
  int min_nnz = matrix->nnz;

  for (int i = 0; i < matrix->nrows; i++) {
    int nnz = matrix->row_ptr[i + 1] - matrix->row_ptr[i];
    max_nnz = (nnz > max_nnz) ? nnz : max_nnz;
    min_nnz = (nnz < min_nnz) ? nnz : min_nnz;
  }

  printf("  Avg non-zeros per row: %.2f\n", avg_nnzPerRow);
  printf("  Min non-zeros in a row: %d\n", min_nnz);
  printf("  Max non-zeros in a row: %d\n", max_nnz);
  printf("\n");
}

int build_coalesced_row_bins(const int *row_ptr, const int rows, int *bin_rows, int warp_size) {
  int bin_nnz_target = warp_size * 2;
  int num_bins = 0;
  int nnz_count = 0;
  bin_rows[0] = 0;

  for (int i = 0; i < rows; i++) {
    int nnz_in_row = row_ptr[i + 1] - row_ptr[i];

    if (nnz_count > 0 && nnz_count + nnz_in_row > bin_nnz_target) {
      num_bins++;
      bin_rows[num_bins] = i;
      nnz_count = 0;
    }
    nnz_count += nnz_in_row;
  }

  if (bin_rows[num_bins] != rows) {
    num_bins++;
    bin_rows[num_bins] = rows;
  }

  return num_bins;
}

// int build_coalesced_row_bins(const int *row_ptr, const int rows, int *bin_rows, int warp_size) {
//   int max_row_per_bin = warp_size * 3;
//   int num_bins = 0;
//   int nnz_count = 0;
//   int target_size = warp_size;
//   bin_rows[0] = 0;

//   for (int i = 0; i < rows; i++) {
//     int nnz_in_row = row_ptr[i + 1] - row_ptr[i];

//     if (nnz_count >= max_row_per_bin || nnz_count + nnz_in_row > max_row_per_bin) {
//       num_bins++;
//       bin_rows[num_bins] = i;
//       nnz_count = 0;
//       target_size = warp_size;
//     }

//     nnz_count += nnz_in_row;

//     if (nnz_count >= warp_size) {
//       if (nnz_count % target_size == 0 || nnz_count >= max_row_per_bin) {
//         num_bins++;
//         bin_rows[num_bins] = i + 1;
//         nnz_count = 0;
//         target_size = warp_size;
//       } else {
//         target_size = ((nnz_count / warp_size) + 1) * warp_size;
//       }
//     }
//   }
//   if (bin_rows[num_bins] != rows) {
//     num_bins++;
//     bin_rows[num_bins] = rows;
//   }

//   return num_bins;
// }