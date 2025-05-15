#ifndef __TEST_UTILS_H__
#define __TEST_UTILS_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WARM_UP 2
#define REP 10

#define TIMER_DEF(n)	 struct timeval temp_1_##n={0,0}, temp_2_##n={0,0}
#define TIMER_START(n)	 gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n)	 gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec-temp_1_##n.tv_sec)*1.e6+(temp_2_##n.tv_usec-temp_1_##n.tv_usec))
#define TIMER_PRINT(n) \
    do { \
        int rk;\
        MPI_Comm_rank(MPI_COMM_WORLD, &rk);\
        if (rk==0) printf("Timer elapsed: %lfs\n", TIMER_ELAPSED(n)/1e6);\
        fflush(stdout);\
        sleep(0.5);\
        MPI_Barrier(MPI_COMM_WORLD);\
    } while (0);

float arithmetic_mean(float* times, int n);

float geometric_mean(float* times, int n);

float calculate_GFlops(int n, float time);

#ifdef __cplusplus
}
#endif

#endif // __TEST_UTILS_H__