#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <sys/time.h>
//#include "multipi.h"

#define N (1<<10)
#define M (1<<10)
#define THREADBLOCKSIZE 1024
#define LENGTH (N*sizeof(point))
#define INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define MARK_TIME(t) gettimeofday(&t, NULL)
#define CALC_TIME(t1, t2) (1.0e6 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec))/(1.0e6)
#define PRINT_TIME(t, s) printf("It took %f seconds to do %s\n",t,s)


typedef struct {
  int num;
  float pi;
} CalcTest;

#define DEBUG False

extern __shared__ float sdat[];

// setup all the threads on a gpu with individual rand seeds
__global__ void
kernel_setup(curandState *states, int d) {
  int i = INDEX;
  curand_init(d, i, 0, &states[i]);
}

// generate M random points for each thread, and count how many have len <= 1.0
__device__ void
d_gen(curandState *globalState) {
  int i = INDEX;
  curandState localState = globalState[i];
  int tid = threadIdx.x;
  for (int s = 0; s < M; s++) {
    float rx = curand_uniform(&localState);
    float ry = curand_uniform(&localState);
    float mag = rx*rx + ry*ry;
    if (mag <= 1.0f) {
      sdat[tid] += 1.0;
    }
  }
  globalState[i] = localState;
  sdat[tid] *= 4.0f;
  sdat[tid] *= 1.0/M;
}

__device__ void
d_count(float *sums) {
  int tid = threadIdx.x;
  for (int i = blockDim.x/2; i > 0; i >>= 1) {
    if (tid < i)
      sdat[tid] += sdat[tid + i];
   __syncthreads();
  }
  __syncthreads();
  if (tid == 0) {
    sdat[0] *= 1.0/blockDim.x;
    sums[blockIdx.x] = sdat[0];
  }
}

// gets the points within a circle of radius 1.0
__global__ void
generate(curandState *globalState, float *sums) {
  d_gen(globalState);
  __syncthreads();
  d_count(sums);
}

CalcTest *
runTest(int num) {
  printf("calculating pi with num %d\n", num);
  CalcTest *t = (CalcTest *)malloc(sizeof(CalcTest));
  t->num = num;

  //printf("MultiGPU Pi is running...\n");
  struct timeval begin, t1, t2;
  //MARK_TIME(begin);

  //MARK_TIME(t1);
  int numDevs = 0;
  cudaGetDeviceCount(&numDevs);
  numDevs = 1;
  //MARK_TIME(t2);
  //PRINT_TIME(CALC_TIME(t1,t2),"get device count");
  //printf("We have access to %d devices\n", numDevs);

  dim3 block, grid;
  block.x = THREADBLOCKSIZE;
  grid.x = (num + THREADBLOCKSIZE - 1)/THREADBLOCKSIZE;
  //printf("grid.x %d\n", grid.x);
  //printf("block.x %d\n", block.x);

  //MARK_TIME(t1);
  float *a[numDevs];
  for (int d = 0; d < numDevs; d++) {
    a[d] = (float *)malloc(grid.x*sizeof(float));
  }
  //MARK_TIME(t2);
  //PRINT_TIME(CALC_TIME(t1,t2),"allocal host arrays");

  // allocate memory on each device
  //printf("allocating memory...");
  //MARK_TIME(t1);
  float *d_a[numDevs];
  curandState *states[numDevs];
  for (int d = 0; d < numDevs; d++) {
    cudaSetDevice(d);
    cudaMalloc(&d_a[d],grid.x*sizeof(float));
    cudaMalloc(&states[d],num*sizeof(curandState));
  }
  //MARK_TIME(t2);
  //PRINT_TIME(CALC_TIME(t1,t2),"allocal cuda arrays");
  //printf("done\n");

  // run the kernel on each device
  //printf("\nrunning kernels...");
  //MARK_TIME(t1);
  for (int d = 0; d < numDevs; d++) {
    cudaSetDevice(d);
    kernel_setup<<<grid, block>>>(states[d],d);
    generate<<<grid, block, block.x*sizeof(float)>>>(states[d], d_a[d]);
  }
  //MARK_TIME(t2);
  //PRINT_TIME(CALC_TIME(t1,t2),"execute kernels");
  //printf("done\n");

  // copy data back to host
  //MARK_TIME(t1);
  for (int d = 0; d < numDevs; d++) {
    cudaSetDevice(d);
    cudaMemcpy(a[d],d_a[d],grid.x*sizeof(float), D2H);
  }
  //MARK_TIME(t2);
  //PRINT_TIME(CALC_TIME(t1,t2),"copy mem back to host");

  // now print the host arrays
  //MARK_TIME(t1);
  int num_print = 2;//grid.x;
  float total = 0.0;
  for (int d = 0; d < numDevs; d++) {
    printf("Values received from device %d:\n",d);
    for (int i = 0; i < grid.x; i++) {
      total += a[d][i];
      if (i < num_print) printf("\ta[%d][%d]: %f\n",d,i,a[d][i]);
    }
  }

  total *= 1.0/(grid.x * numDevs);
  printf("Estimate of pi: %f\n",total);
  t->pi = total;

  for (int d = 0; d < numDevs; d++) {
    free(a[d]);
    cudaFree(d_a[d]);
  }
  //MARK_TIME(t2);
  //PRINT_TIME(CALC_TIME(t1,t2),"print results and clean up");

  //PRINT_TIME(CALC_TIME(begin,t2),"execute the entire program");

  return t;
}

int
main(void) {

  /*
  printf("MultiGPU Pi is running...\n");
  struct timeval begin, t1, t2;
  MARK_TIME(begin);

  MARK_TIME(t1);
  int numDevs = 0;
  cudaGetDeviceCount(&numDevs);
  numDevs = 2;
  MARK_TIME(t2);
  PRINT_TIME(CALC_TIME(t1,t2),"get device count");
  printf("We have access to %d devices\n", numDevs);

  dim3 block, grid;
  block.x = THREADBLOCKSIZE;
  grid.x = (N + THREADBLOCKSIZE - 1)/THREADBLOCKSIZE;
  printf("grid.x %d\n", grid.x);
  printf("block.x %d\n", block.x);

  MARK_TIME(t1);
  float *a[numDevs];
  for (int d = 0; d < numDevs; d++) {
    a[d] = (float *)malloc(grid.x*sizeof(float));
  }
  MARK_TIME(t2);
  PRINT_TIME(CALC_TIME(t1,t2),"allocal host arrays");

  // allocate memory on each device
  printf("allocating memory...");
  MARK_TIME(t1);
  float *d_a[numDevs];
  curandState *states[numDevs];
  for (int d = 0; d < numDevs; d++) {
    cudaSetDevice(d);
    cudaMalloc(&d_a[d],grid.x*sizeof(float));
    cudaMalloc(&states[d],N*sizeof(curandState));
  }
  MARK_TIME(t2);
  PRINT_TIME(CALC_TIME(t1,t2),"allocal cuda arrays");
  printf("done\n");

  // run the kernel on each device
  printf("\nrunning kernels...");
  MARK_TIME(t1);
  for (int d = 0; d < numDevs; d++) {
    cudaSetDevice(d);
    kernel_setup<<<grid, block>>>(states[d],d);
    generate<<<grid, block, block.x*sizeof(float)>>>(states[d], d_a[d]);
  }
  MARK_TIME(t2);
  PRINT_TIME(CALC_TIME(t1,t2),"execute kernels");
  printf("done\n");

  // copy data back to host
  MARK_TIME(t1);
  for (int d = 0; d < numDevs; d++) {
    cudaSetDevice(d);
    cudaMemcpy(a[d],d_a[d],grid.x*sizeof(float), D2H);
  }
  MARK_TIME(t2);
  PRINT_TIME(CALC_TIME(t1,t2),"copy mem back to host");

  // now print the host arrays
  MARK_TIME(t1);
  int num_print = grid.x;
  float total = 0.0;
  for (int d = 0; d < numDevs; d++) {
    printf("Values received from device %d:\n",d);
    for (int i = 0; i < grid.x; i++) {
      total += a[d][i];
      if (i < num_print) printf("\ta[%d][%d]: %f\n",d,i,a[d][i]);
    }
  }

  total *= 1.0/(grid.x * numDevs);
  printf("Estimate of pi: %f\n",total);

  for (int d = 0; d < numDevs; d++) {
    free(a[d]);
    cudaFree(d_a[d]);
  }
  MARK_TIME(t2);
  PRINT_TIME(CALC_TIME(t1,t2),"print results and clean up");

  PRINT_TIME(CALC_TIME(begin,t2),"execute the entire program");
  */

  int numTests = 10;
  CalcTest *tests[numTests];
  for (int i = 0; i < numTests; i++) {
    tests[i] = runTest(N);
  }

  FILE *fp;
  fp = fopen("results_multi.txt","w");
  fprintf(fp, "N\tpi\n");
  for (int i = 0; i < numTests; i++) {
    fprintf(fp, "%d\t%f\n", tests[i]->num, tests[i]->pi);
  }
  fclose(fp);

  return 0;
}

