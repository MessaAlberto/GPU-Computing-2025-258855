#ifndef __MTX_UTILS_H__
#define __MTX_UTILS_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef struct CSR_matrix {
  int nrows;
  int ncols;
  int nnz;
  int* row_ptr;
  int* col_idx;
  double* values;
} CSR_matrix;

void COO_to_CSR(int nrows, int ncols, int nnz,
                 const int* row_coo, const int* col_coo,
                 const double* values_coo,
                 CSR_matrix* matrix);

void read_CSR_mtx(const char* filename, CSR_matrix* matrix);

double* init_RandVector(int ncols);

#endif // __MTX_UTILS_H__