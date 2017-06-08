#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <sys/time.h>

#define NN (1<<10) // number of seeds
#define MM (1<<20) // number of samples per seed
#define THREADBLOCKSIZE 1024
#define LENGTH (N*sizeof(float))
#define INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define MARK_TIME(t) gettimeofday(&t, NULL)
#define CALC_TIME(t1, t2) (1.0e6 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec))/(1.0e6)
#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice

extern __shared__ float sdat[];

/*
Calculates an estimate for pi for each thread. Uses m random numbers
generated using the globalState random number generator
*/
__device__ void d_gen(curandState *globalState, int m) {
  int i = INDEX;
  curandState localState = globalState[i];

  int spt = m; //number of samples per thread
  int tid = threadIdx.x;
  int count = 0;
  for (int s = 0; s < spt; s++) {
    float rx = curand_uniform(&localState);
    float ry = curand_uniform(&localState);
    float mag = rx*rx + ry*ry;
    if (mag <= 1.0f) {
      count += 1;
      sdat[tid] += 1.0;
    }
  }

  globalState[i] = localState;
  sdat[tid] *= 1.0/spt;
}

/*
Creates an estimate of pi for an entire block. Sums the
individual thread estimates and then divides by the number of threads
in a block.
*/
__device__ void d_count(float *sums) {
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

/*
Actuallyt generates the estimates for pi for each block once the random
number generators are correctly set up.
*/
__global__ void generate(curandState *globalState, float *sums, int m) {
  d_gen(globalState, m);
  __syncthreads();
  d_count(sums);
  __syncthreads();
}

/*
Sets up the random number generators for the blocks. This step must be
called before the pi estimates are generated.
*/
__global__ void kernel_setup(curandState *states) {
  int i = INDEX;
  curand_init(0, i, 0, &states[i]);
}

/*
Actually generates the estimate of pi.
*/
int main(int argc, char *argv[]) {
 
  int N,M;

  if (argc == 3) {
    N = 1 << atoi(argv[1]);
    M = 1 << atoi(argv[2]);
    printf("N: %d, M: %d\n",N, M);
  } else {
    N = NN;
    M = MM;
  }


  printf("sizeof curandState %d\n", sizeof(curandState));
  struct timeval begin, t1, t2; //, bs1, bs2;
  MARK_TIME(begin);
  printf("starting pi calc...\n");

  dim3 block, grid;
  block.x = THREADBLOCKSIZE;
  grid.x = (N + THREADBLOCKSIZE - 1)/THREADBLOCKSIZE;
  printf("grid.x %d\n", grid.x);
  printf("block.x %d\n", block.x);


  MARK_TIME(t1);
  printf("mallocing on host and device\n");
  float *p, *d_p;
  p = (float *)malloc(grid.x*sizeof(float));
  MARK_TIME(t2);
  printf("it took %f seconds to allocate p...\n", CALC_TIME(t1, t2));
  cudaMalloc(&d_p,grid.x*sizeof(float));
  MARK_TIME(t1);
  printf("it took %f seconds to allocate d_p...\n", CALC_TIME(t2, t1));

  curandState *states;
  cudaMalloc(&states, N*sizeof(curandState));
  MARK_TIME(t2);
  printf("it took %f seconds to allocate states...\n", CALC_TIME(t1, t2));

  printf("running kernel_setup...");
  MARK_TIME(t1);
  kernel_setup<<<grid, block>>>(states);
  MARK_TIME(t2);
  printf("done\n");
  printf("it took %f seconds to execute kernel_setup...\n", CALC_TIME(t1, t2));

  printf("running generate...");
  MARK_TIME(t1);
  generate<<<grid, block, block.x*sizeof(float)>>>(states, d_p, M);
  MARK_TIME(t2);
  printf("done\n");
  printf("it took %f seconds to execute generate...\n", CALC_TIME(t1, t2));

  printf("starting cuda memcpy...\n");
  MARK_TIME(t1);
  cudaMemcpy(p, d_p, grid.x*sizeof(float), cudaMemcpyDeviceToHost);
  MARK_TIME(t2);
  printf("done\n");
  printf("it took %f seconds to memcpy to host...\n", CALC_TIME(t1, t2));

  MARK_TIME(t1);
  int num_print = grid.x;
  float total = 0.0;
  for (int i = 0; i < num_print; i++) {
    //printf("i:%d\tsum %f\n",i,p[i]);
    total += p[i];
  }

  float pi = 4.0 * total / grid.x;
  printf("pi estimate: %f\n", pi);

  printf("cleaning up\n");
  cudaFree(states);
  cudaFree(d_p);
  free(p);
  MARK_TIME(t2);
  printf("it took %f seconds to calc total and clean up\n", CALC_TIME(t1,t2));
  printf("\nThe total execution time of this program was %f seconds\n", CALC_TIME(begin,t2));

  FILE *fp = fopen("gpu_results.dat", "a");
  fprintf(fp, "%d %d %.10f\n",N,M,pi);
  fclose(fp);

  return 0;
}
