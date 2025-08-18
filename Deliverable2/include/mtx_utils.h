#ifndef __MTX_UTILS_H__
#define __MTX_UTILS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef struct COO_matrix {
  int nrows;
  int ncols;
  int nnz;
  int* row_idx;
  int* col_idx;
  double* values;
} COO_matrix;

typedef struct CSR_matrix {
  int nrows;
  int ncols;
  int nnz;
  int* row_ptr;
  int* col_idx;
  double* values;
} CSR_matrix;

void read_COO_mtx(const char* filename, COO_matrix* matrix);

void COO_to_CSR(COO_matrix* coo_matrix, CSR_matrix* matrix);

void init_RandVector(double* vec, int n);

void print_mtx_stats(CSR_matrix* matrix);

int build_coalesced_row_bins(const int* row_ptr, const int rows, int* bin_rows, int warp_size);

void classify_rows(int* row_ptr, int rows, int* short_rows, int* long_rows, int* short_count, int* long_count, int threshold);

#ifdef __cplusplus
}
#endif
#endif // __MTX_UTILS_H__