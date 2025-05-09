#include <iostream>
#include <math.h>

#define WARM_UP 2
#define REP 10

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { 
    y[i] = x[i] + y[i];
  }
}

float arithmetic_mean(float *vector, int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += vector[i];
    }
    return sum / size;
}

int main(void)
{  
  float times[REP] = {0};
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int N = 1<<20;
  float *x, *y;
  
  // Allocate Unified Memory accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int threadsPerBlock = 512;
  int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  printf("Blocks: %d, Threads per block: %d\n", blocks, threadsPerBlock);

  for (int i = -WARM_UP; i < REP; i++) {
    cudaEventRecord(start);
    add<<<blocks, threadsPerBlock>>>(N, x, y);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int j = 0; j < N; j++)
    {
      maxError = fmax(maxError, fabs(y[j]-3.0f));
      if (y[j] != 3.0)
      {
        printf("err in %d: %f\n", j, y[j]);
        return 1;
      }
      y[j] = 2.0f;
    }

    if (i >= 0) {
      cudaEventElapsedTime(&times[i], start, stop);
    } else {
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("Warm-up %f ms\n", milliseconds);
    }
  }
  
  float mean = arithmetic_mean(times, REP);
  float gflops = 2.0 * N / (mean * 1e6); // GFlops
  printf("Time: %f ms, GFlops: %f\n", mean, gflops);
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(x);
  cudaFree(y);
}