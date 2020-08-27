#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>

#define BLOCK_SIZE 60
#define NUM_SM 30

__device__ __inline__ uint32_t get_smid(){
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// There is no mechanism for waiting non-processed blocks
// modification for vectorAdd
__global__ void TransformedKernel(const float *A, const float *B, 
                                  float *C, int numElements,
                                  int grid_size, int *block_index, 
                                  int *max_blocks, volatile int *concurrent_blocks){

    int smid = get_smid();
    __shared__ int logicalBlockIdx;
    __shared__ int physicalBlockIdx;
    
    if(threadIdx.x == 0){
        physicalBlockIdx = atomicAdd(&(block_index[smid+1]), 1);    // physicalBlockIdx is always 0 (cause, it gets old val)
    }
       
    __syncthreads();
    while(physicalBlockIdx < *max_blocks){                      // physicalBlockIdx: 0, *max_blocks: 1
        if(threadIdx.x == 0){
            logicalBlockIdx = atomicAdd(&(block_index[0]),1);   //Array `blcok_index' is shared by all other kernels
            *concurrent_blocks = logicalBlockIdx;               //logicalBlockIdx is not shared among kernels
        }
        __syncthreads();
        if(logicalBlockIdx >= grid_size){
            break;
        }
        // ORIGINAL CODE
        // You should not use return value of atomic operations in kernel code, because the return vaule is random.
        // for atomic operations, there is no guarantee the order of performing operations. Thus, there is no guarantee of 
        // getting the reproducible return value. 
        /*
        if(threadIdx.x == 0)
            d_test[blockIdx.x] = *concurrent_blocks;
        */
        // Vector Add Kernel Code
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < numElements){
            C[i] = A[i] + B[i];
        }
        //atomicAdd(&(d_test[*concurrent_blocks]), 1);
        __syncthreads();
    }
    
    if(threadIdx.x == 0){
        atomicSub(&(block_index[smid+1]), 1);
    }
    
}


__global__ void AtomicArrayTest(int* block_index){
    if(threadIdx.x == 0)
    {
        int smid = get_smid();
        atomicAdd(&(block_index[smid]), 1);
        //__syncthreads();
    }    
}

int main(){
    // PARAMETER SETTING TO USE PERSISTENT THREAD
    int grid_size = 30;                     // number of Blocks in the GRID
    int nThreads = 1;                       // number of Threads in the TheadBlock

    const int num_sm = 30;                  // number of SM of TITAN XP
    int *block_index;                       // device pointer representing physicalBlockIdx of 
                                            //                             specific SM
    cudaMalloc((void**)&block_index, sizeof(int)*(num_sm + 1));
    cudaMemset(block_index, 0, sizeof(int)*(num_sm + 1));

    int host_max_blocks = 1;
    int *max_blocks;                        // Maximum number of CTA in a SM
    cudaMalloc((void**)&max_blocks, sizeof(int));
    cudaMemset(max_blocks, 0, sizeof(int));
    cudaMemcpy(max_blocks, &host_max_blocks, sizeof(int), cudaMemcpyHostToDevice);
    
    volatile int *concurrent_blocks;
    cudaMalloc((void**)&concurrent_blocks, sizeof(int));

    //int *test;
    //test = (int*)malloc(sizeof(int)*grid_size);
    //int *d_test;
    //cudaMalloc((void**)&d_test, sizeof(int)*grid_size);
    //cudaMemset(d_test, 0, sizeof(int)*grid_size);

    //int numElements = 5000;
    int numElements = 10000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i){
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;  
    // If numElements=5000 -> blocksPerGrid: 20
    // If numElements=10000 -> blocksPerGrid: 40
    grid_size = blocksPerGrid;
    nThreads = threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock); 
    // Transformed Vector Add Kernel
    TransformedKernel<<<grid_size, nThreads>>>(d_A, d_B,
                                               d_C, numElements,
                                               grid_size, 
                                               block_index, max_blocks,
                                               concurrent_blocks);
    int *host_block_index;
    host_block_index = (int*)malloc(sizeof(int)*(num_sm + 1));
    
    cudaMemcpy(host_block_index, block_index, sizeof(int)*(num_sm + 1), cudaMemcpyDeviceToHost);
    for(int i = 0; i < num_sm + 1; i++)
    {
        printf("%d \n", host_block_index[i]);
    }
    
    int *h_result;
    h_result = (int*)malloc(sizeof(int)*1);

    cudaMemcpy(h_result, d_C, sizeof(int)*1, cudaMemcpyDeviceToHost);
    printf("[Vector addition of %d elements] h_result: %d \n", numElements, *h_result);
    //cudaMemcpy(test, d_test, sizeof(int)*grid_size, cudaMemcpyDeviceToHost);

    //for(int i = 0; i < grid_size ; i++)
    //    printf("test %d  %d\n",i, test[i]);
    


    return 0;
}
