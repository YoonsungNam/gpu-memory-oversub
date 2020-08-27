#include <iostream>
#include <math.h>

// CUDA kernel to add elements of two arrays
__global__ void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

void printPtrInfo(char c, cudaPointerAttributes attributes) {
    printf("\n\ninfo of %c\n", c);
    printf(" Memory type   %i\n",attributes.memoryType);
    printf(" Type          %i\n",attributes.type);
    printf(" Device        %d\n",attributes.device);
    printf(" isManaged     %d\n",attributes.isManaged);
}

int main(void)
{
    int N = 1<<30;
    float *x, *y;

    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMemAdvise(x, N*sizeof(float), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    // get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    printf("0 Unregistered\n1 Host\n2 Device\n3 Managed\n");
    
 
    cudaPointerAttributes attributes;
    printf("Pointer info\n");

    cudaPointerGetAttributes(&attributes, x);
    printPtrInfo('x', attributes);
    
    cudaPointerGetAttributes(&attributes, y);
    printPtrInfo('y', attributes);
    

    // Launch kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
