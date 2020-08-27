// This example code is from "http://blog.naver.com/PostView.nhn?blogId=sogangori&logNo=220580711355"
// Reduction code to find the max value from array

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void FindMaxCUDA(int *src, int *max_dst, int length){
    extern __shared__ int sm[];     // Allocating the shared memory dynamically

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    if(i < length){
        sm[tid] = src[i];
        __syncthreads();

        for(int s = 1; s < blockDim.x; s *= 2){
            if (tid%(2*s)==0){
                if(sm[tid] < sm[tid+s]) sm[tid] = sm[tid+s];
            }
            __syncthreads();
        }

        if(tid==0) max_dst[0] = sm[0];  // If you want to get max value using `divide-and-conquer`, 
                                        // the index for max_dst should be changed to `blockIdx.x`.
                                        // And you may call the kernel once again for finding the max value from max_dst array.
    }
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = {1,2,5,4,3};
    int max[1] = {0};

    int *src_d;
    int *max_d;
    cudaMalloc((void**)&src_d, sizeof(int)*arraySize);
    cudaMalloc((void**)&max_d, sizeof(int)*1);
    cudaMemcpy(src_d, a, sizeof(int)*arraySize, cudaMemcpyHostToDevice);

    // NOTE: the third parameter is used for dynamically allocating shared memory in the kernel. 
    FindMaxCUDA<<<1, arraySize, arraySize*sizeof(int)>>>(src_d, max_d, arraySize);  

    cudaMemcpy(max, max_d, sizeof(int), cudaMemcpyDeviceToHost);

    printf("max = %d\n", max[0]);

   return 0;
} 
                        
