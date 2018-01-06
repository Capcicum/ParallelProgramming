// Reduction example for parallel programming reading course
//
// Auther: Frederik Andersen
//
// Setup of code and comment is modified from CUDA example addKernel
// Reduce kernel is taken from http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

cudaError_t reduce(int *h_in, int *h_out, unsigned int size);
void createArray(const int size, int* array);
void createResult(int* array, int* result, const int size);


// Reducion 
__global__ void reduce_kernel(int * d_out, int * d_in)
{
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	sdata[tid] = d_in[i] + d_in[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem

	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}
}

int main()
{
	const int arraySize = 256;
	int h_in[arraySize] = { 0 };
	int h_out[1] = { 0 };
	int serialResult[1] = { 0 };

	createArray(arraySize, h_in);
	createResult(h_in, serialResult, arraySize);

    // Add vectors in parallel.
    cudaError_t cudaStatus = reduce(h_in, h_out, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("Reduce of %d elements resulted in = {%d}\n", arraySize, h_out[0]);
	if (h_out[0] == serialResult[0])
		printf("Test Passed\n");
	else
		printf("Test Failed, result should be %d but is %d\n", serialResult[0], h_out[0]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.


    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	int lol;
	std::cin >> lol;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t reduce(int *h_in, int *h_out,  unsigned int size)
{
    int *d_in = 0;
    int *d_out = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&d_in, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_out, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_in, h_in, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(d_out, h_out, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	reduce_kernel <<<1, size, size*sizeof(int)>>>(d_out, d_in);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "reduce_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reduce_kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(d_out);
    cudaFree(d_in);
    
    return cudaStatus;
}

void createArray(const int size, int* array)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = i;
	}
}

void createResult(int* array, int* result, const int size)
{
	for (int i = 0; i < size; i++)
	{
		result[0] += array[i];
	}
}