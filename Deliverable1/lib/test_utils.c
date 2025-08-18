#include "../include/test_utils.h"

float arithmetic_mean(float* times, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += times[i];
  }
  return sum / n;
}

float geometric_mean(float* times, int n) {
  float product = 1.0f;
  for (int i = 0; i < n; i++) {
    product *= times[i];
  }
  return pow(product, 1.0f / n);
}

float calculate_GFlops(int n, float time) {
  float res = (2.0f * n) / (time * 1e9); // GFlops
  return res;
}