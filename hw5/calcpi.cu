#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <sys/time.h>

#define NN (1<<5) // number of seeds
#define MM (1<<10) // number of samples per seed
#define THREADBLOCKSIZE 1024
#define LENGTH (N*sizeof(float))
#define INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define MARK_TIME(t) gettimeofday(&t, NULL)
#define CALC_TIME(t1, t2) (1.0e6 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec))/(1.0e6)
#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice

typedef struct {
  float x, y;
} point;

extern __shared__ float sdat[];

__device__ void d_gen(curandState *globalState, int m) {
  int i = INDEX;
  curandState localState = globalState[i];

  int spt = m; //M; //number of samples per thread
  int tid = threadIdx.x;
  //int bid = blockIdx.x;
  int count = 0;
  //printf("tid:%d\tbid:%d\ti:%d\n",tid,bid,i);
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
  //sdat[tid] *= 4.0f;
  sdat[tid] *= 1.0/spt;

}

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

__global__ void generate(curandState *globalState, float *sums, int m) {
  d_gen(globalState, m);
  __syncthreads();
  d_count(sums);
  __syncthreads();
}

__global__ void kernel_setup(curandState *states) {
  int i = INDEX;
  curand_init(0, i, 0, &states[i]);
}

int main(int argc, char *argv[]) {
 
  int N,M;

  if (argc > 1) {
    if (argc != 3) {
      printf("wrong number of args. exiting\n");
      return -1;
    }
    N = 1 << atoi(argv[1]);
    M = 1 << atoi(argv[2]);
    printf("N: %d, M: %d\n",N, M);
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

  printf("inside: %f\n", total);
  printf("ratio: %f\n", 1.0f*total/grid.x);
  printf("pi estimate: %f\n", pi);

  printf("sizeof(float) %d\n",sizeof(float));
  printf("block (x,y,z): (%d,%d,%d)\n",block.x,block.y,block.z);
  printf("grid  (x,y,z): (%d,%d,%d)\n",grid.x, grid.y, grid.z);

  printf("cleaning up\n");
  cudaFree(states);
  cudaFree(d_p);
  free(p);
  MARK_TIME(t2);
  printf("it took %f seconds to calc total and clean up\n", CALC_TIME(t1,t2));
  printf("\nThe total execution time of this program was %f seconds\n", CALC_TIME(begin,t2));

FILE *fp = fopen("test.dat", "a");
//fseek(fp, SEEK_END);
fprintf(fp, "%d %d %.10f\n",N,M,pi);
fclose(fp);

}
