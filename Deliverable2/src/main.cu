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
  #define KERNEL_PARAMS num_bins, matrix.row_ptr, matrix.col_idx, matrix.values, vec, result, bin_rows

#elif SELECT_KERNEL == 4
  #define KERNEL SpMV_Hybrid
  #define KERNEL_NAME "SpMV_Hybrid"
  #define KERNEL_PARAMS                                                                   \
    matrix.nrows, matrix.row_ptr, matrix.col_idx, matrix.values, vec, result, short_rows, \
    long_rows, num_short_rows, num_short_rows_padded, num_long_rows
#else
  #error "SELECT_KERNEL must be 1..4"
#endif

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

#define CUDA_CHECK_KERNEL()                                      \
  do {                                                           \
    cudaError_t err = cudaGetLastError();                        \
    if (err != cudaSuccess) {                                    \
      fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err)); \
      return 1;                                                  \
    }                                                            \
  } while (0)

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))

// ===================== Kernels =====================

// 1) CSR-Scalar: 1 thread per riga
__global__ __launch_bounds__(256, 2)
void SpMV_OneThreadPerRow(const int rows,
                          const int *__restrict__ row_ptr,
                          const int *__restrict__ col_idx,
                          const double *__restrict__ values,
                          const double *__restrict__ vec,
                          double *__restrict__ result)
{
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows;
       row += blockDim.x * gridDim.x)
  {
    const int start = row_ptr[row];
    const int end   = row_ptr[row + 1];

    double sum = 0.0;
    int j = start;
    for (; j + 3 < end; j += 4) {
      sum += values[j]     * __ldg(&vec[col_idx[j]]);
      sum += values[j + 1] * __ldg(&vec[col_idx[j + 1]]);
      sum += values[j + 2] * __ldg(&vec[col_idx[j + 2]]);
      sum += values[j + 3] * __ldg(&vec[col_idx[j + 3]]);
    }
    for (; j < end; ++j) {
      sum += values[j] * __ldg(&vec[col_idx[j]]);
    }
    result[row] = sum;
  }
}

// 2) CSR-Vector: 1 warp per riga
__global__ __launch_bounds__(256, 2)
void SpMV_OneWarpPerRow(const int rows,
                        const int *__restrict__ row_ptr,
                        const int *__restrict__ col_idx,
                        const double *__restrict__ values,
                        const double *__restrict__ vec,
                        double *__restrict__ result)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warps_per_grid = (gridDim.x * blockDim.x) / WARP_SIZE;

  for (int warp = tid / WARP_SIZE; warp < rows; warp += warps_per_grid) {
    const int start = row_ptr[warp];
    const int end   = row_ptr[warp + 1];

    double sum = 0.0;
    #pragma unroll 4
    for (int j = start + lane; j < end; j += WARP_SIZE) {
      sum += values[j] * __ldg(&vec[col_idx[j]]);
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) result[warp] = sum;
  }
}

// 3) Bins coalesced: warp che percorre tutte le righe del bin (niente atomiche)
__global__ __launch_bounds__(256, 2)
void SpMV_coalescedBins(const int num_bins,
                        const int *__restrict__ row_ptr,
                        const int *__restrict__ col_idx,
                        const double *__restrict__ values,
                        const double *__restrict__ vec,
                        double *__restrict__ result,
                        const int *__restrict__ bin_rows)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warps_per_grid = (gridDim.x * blockDim.x) / WARP_SIZE;

  for (int bw = tid / WARP_SIZE; bw < num_bins; bw += warps_per_grid) {
    const int row_start = bin_rows[bw];
    const int row_end   = bin_rows[bw + 1];

    for (int row = row_start; row < row_end; ++row) {
      const int start = row_ptr[row];
      const int end   = row_ptr[row + 1];

      double sum = 0.0;
      #pragma unroll 4
      for (int j = start + lane; j < end; j += WARP_SIZE) {
        sum += values[j] * __ldg(&vec[col_idx[j]]);
      }

      sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
      sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
      sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
      sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
      sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

      if (lane == 0) result[row] = sum;
    }
  }
}

// 4) Hybrid: short -> 1TPR, long -> 1 warp
__global__ __launch_bounds__(256, 2)
void SpMV_Hybrid(const int rows,
                 const int *__restrict__ row_ptr,
                 const int *__restrict__ col_idx,
                 const double *__restrict__ values,
                 const double *__restrict__ vec,
                 double *__restrict__ result,
                 const int *__restrict__ short_rows,
                 const int *__restrict__ long_rows,
                 const int num_short_rows,
                 const int /*num_short_rows_padded*/,
                 const int num_long_rows)
{
  const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & (WARP_SIZE - 1);

  // short rows: grid-stride 1TPR
  for (int i = tid; i < num_short_rows; i += gridDim.x * blockDim.x) {
    const int row   = short_rows[i];
    const int start = row_ptr[row];
    const int end   = row_ptr[row + 1];

    double sum = 0.0;
    int j = start;
    for (; j + 3 < end; j += 4) {
      sum += values[j]     * __ldg(&vec[col_idx[j]]);
      sum += values[j + 1] * __ldg(&vec[col_idx[j + 1]]);
      sum += values[j + 2] * __ldg(&vec[col_idx[j + 2]]);
      sum += values[j + 3] * __ldg(&vec[col_idx[j + 3]]);
    }
    for (; j < end; ++j) {
      sum += values[j] * __ldg(&vec[col_idx[j]]);
    }
    result[row] = sum;
  }

  // __syncthreads();

  // long rows: grid-stride in unità warp
  const int warps_per_grid = (gridDim.x * blockDim.x) / WARP_SIZE;
  for (int w = (tid / WARP_SIZE); w < num_long_rows; w += warps_per_grid) {
    const int row   = long_rows[w];
    const int start = row_ptr[row];
    const int end   = row_ptr[row + 1];

    double sum = 0.0;
    #pragma unroll 4
    for (int j = start + lane; j < end; j += WARP_SIZE) {
      sum += values[j] * __ldg(&vec[col_idx[j]]);
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) result[row] = sum;
  }
}

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

  // Advise CUDA on memory usage
  CUDA_CHECK(cudaMemAdvise(matrix.values, matrix.nnz * sizeof(double),
                           cudaMemAdviseSetPreferredLocation, device));
  CUDA_CHECK(cudaMemAdvise(matrix.col_idx, matrix.nnz * sizeof(int),
                           cudaMemAdviseSetPreferredLocation, device));
  CUDA_CHECK(cudaMemAdvise(matrix.row_ptr, (matrix.nrows + 1) * sizeof(int),
                           cudaMemAdviseSetReadMostly, device));
  CUDA_CHECK(cudaMemAdvise(vec, matrix.ncols * sizeof(double),
                           cudaMemAdviseSetReadMostly, device));

  // Prefetch data to the GPU
  CUDA_CHECK(cudaMemPrefetchAsync(matrix.row_ptr, (matrix.nrows + 1) * sizeof(int), device));
  CUDA_CHECK(cudaMemPrefetchAsync(matrix.col_idx, matrix.nnz * sizeof(int), device));
  CUDA_CHECK(cudaMemPrefetchAsync(matrix.values, matrix.nnz * sizeof(double), device));
  CUDA_CHECK(cudaMemPrefetchAsync(vec,            matrix.ncols * sizeof(double), device));
  CUDA_CHECK(cudaMemPrefetchAsync(result,         matrix.nrows * sizeof(double), device));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Based on the selected kernel, set the number of blocks and threads
  int gridDim = 0;

#if SELECT_KERNEL == 1
  // unità di lavoro = righe
  const long long total_threads = matrix.nrows;
  gridDim = (int)MIN( (total_threads + BLOCK_DIM - 1) / (long long)BLOCK_DIM,
                           (long long)SM * 8 );

#elif SELECT_KERNEL == 2
  // unità di lavoro = righe in unità warp
  const long long total_warps = matrix.nrows; // 1 warp per riga
  const long long total_threads = total_warps * WARP_SIZE;
  gridDim = (int)MIN( (total_threads + BLOCK_DIM - 1) / (long long)BLOCK_DIM,
                           (long long)SM * 8 );

#elif SELECT_KERNEL == 3
  // costruiamo i bin
  int *host_bin_rows = (int*)malloc((matrix.nrows + 1) * sizeof(int));
  int num_bins = build_coalesced_row_bins(matrix.row_ptr, matrix.nrows, host_bin_rows, WARP_SIZE);
  int *bin_rows = nullptr;
  CUDA_CHECK(cudaMallocManaged(&bin_rows, (num_bins + 1) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(bin_rows, host_bin_rows, (num_bins + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  free(host_bin_rows);
  CUDA_CHECK(cudaMemPrefetchAsync(bin_rows, (num_bins + 1) * sizeof(int), device));
  CUDA_CHECK(cudaDeviceSynchronize());

  const long long total_warps = num_bins; // 1 warp per bin attivo
  const long long total_threads = total_warps * WARP_SIZE;
  gridDim = (int)MIN( (total_threads + BLOCK_DIM - 1) / (long long)BLOCK_DIM,
                           (long long)SM * 8 );

#elif SELECT_KERNEL == 4
  // short/long split
  int *host_short_rows = (int*)malloc(matrix.nrows * sizeof(int));
  int *host_long_rows  = (int*)malloc(matrix.nrows * sizeof(int));
  int num_short_rows = 0, num_long_rows = 0;

  classify_rows(matrix.row_ptr, matrix.nrows,
                host_short_rows, host_long_rows,
                &num_short_rows, &num_long_rows,
                WARP_SIZE * 2);

  int *short_rows = nullptr, *long_rows = nullptr;
  if (num_short_rows > 0) {
    CUDA_CHECK(cudaMallocManaged(&short_rows, num_short_rows * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(short_rows, host_short_rows, num_short_rows * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemPrefetchAsync(short_rows, num_short_rows * sizeof(int), device));
  } else {
    CUDA_CHECK(cudaMallocManaged(&short_rows, sizeof(int))); // dummy
  }

  if (num_long_rows > 0) {
    CUDA_CHECK(cudaMallocManaged(&long_rows, num_long_rows * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(long_rows, host_long_rows, num_long_rows * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemPrefetchAsync(long_rows, num_long_rows * sizeof(int), device));
  } else {
    CUDA_CHECK(cudaMallocManaged(&long_rows, sizeof(int))); // dummy
  }

  free(host_short_rows);
  free(host_long_rows);
  CUDA_CHECK(cudaDeviceSynchronize());

  const int num_short_rows_padded = ((num_short_rows + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  const long long total_threads_ll = (long long)num_short_rows + (long long)num_long_rows * WARP_SIZE;
  gridDim = (int)MIN( (total_threads_ll + BLOCK_DIM - 1) / (long long)BLOCK_DIM,
                           (long long)SM * 8 );
#endif

  printf("Using kernel: \t\t%s\n", KERNEL_NAME);
  printf("Using matrix: \t\t%s\n", argv[1]);
  printf("Using block size: \t%d\n\n", BLOCK_DIM);
  print_mtx_stats(&matrix);
  fflush(stdout);

  // Warm up
  for (int i = 0; i < WARM_UP; ++i) {
    KERNEL<<<gridDim, BLOCK_DIM>>>(KERNEL_PARAMS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_KERNEL();
  }

  // Benchmarking phase
  for (int i = 0; i < REP; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    KERNEL<<<gridDim, BLOCK_DIM>>>(KERNEL_PARAMS);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK_KERNEL();

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    times[i] = ms * 1e-3f; // Convert to seconds
  }

  // Calculate results
  const float meanTime = arithmetic_mean(times, REP); // sec
  const float flopCount = 2.0f * (float)matrix.nnz;
  const float gflops = calculate_GFlops(flopCount, meanTime);

  // Bandwidth calculation based on the worst-case scenario
  const int Bd = sizeof(double); // 8 bytes
  const int Bi = sizeof(int);    // 4 bytes

  // Accessi "tipici" CSR: row_ptr (per riga), col_idx/values (per nnz), vec (per nnz), result (per riga)
  size_t readBytes   = (size_t)matrix.nrows * (Bi + Bi)   // row_ptr[row], row_ptr[row+1]
                     + (size_t)matrix.nnz * (Bi + Bd + Bi); // col_idx, values, vec
  size_t writeBytes  = (size_t)matrix.nrows * Bd;         // result[row]
#if SELECT_KERNEL == 3
  // niente atomic -> nessuna write per nnz
  // letture extra già coperte da CSR standard (il vecchio while è stato rimosso)
#elif SELECT_KERNEL == 4
  // short_rows/long_rows (letture di metadati) trascurabili rispetto a nnz; se vuoi contarle:
  // readBytes += (size_t)num_short_rows * Bi + (size_t)num_long_rows * Bi;
#endif
  const size_t totalBytes = readBytes + writeBytes;
  const float bandwidthGBs = (float)totalBytes / (meanTime * 1e9f);

  // --- Somma del risultato (spostiamo su host una sola volta, alla fine)
  CUDA_CHECK(cudaMemPrefetchAsync(result, matrix.nrows * sizeof(double), cudaCpuDeviceId));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int r = 0; r < matrix.nrows; ++r) finalResult += result[r];

  printf("Sum of resulting vector: %f\n", finalResult);
  printf("Mean time: %f ms\n", meanTime * 1e3f);
  printf("GFlops: %f\n", gflops);
  printf("Bandwidth: %f GB/s\n", bandwidthGBs);

  // --- Cleanup
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
