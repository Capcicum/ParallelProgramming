// Blelloch scan example for parallel programming reading course
//
// Auther: Frederik Andersen
//
// Setup of code and comment is modified from CUDA example addWithCuda
// Scan kernel is taken from https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define COMPUTE_BIN(n) n && 0x01

cudaError_t histogram(int *h_in, int *h_out, unsigned int bins, unsigned int size);
void createArray(const int size, int* array);
void serialVersion(int* h_in, int* h_out, unsigned int bins, unsigned int size);
bool checkResults(int* array1, int* array2, unsigned int size);


// Blelloch  scan 
__global__ void naive_histo_kernel(int *d_bins, const int *d_in, const int BIN_COUNT)
{
	int myId = threadIdx.x;
	int myItem = d_in[myId];
	int myBin = COMPUTE_BIN(myItem);
	atomicAdd(&(d_bins[myBin]), 1);
}

int main()
{
	const int arraySize = 256;
	const int bins = 2;
	int h_in[arraySize] = { 0 };
	int h_out[bins] = { 0 };
	int serialResult[bins] = { 0 };

	createArray(arraySize, h_in);

	// Scan in serial.
	serialVersion(h_in, serialResult, bins, arraySize);

	// Scan in parallel.
	cudaError_t cudaStatus = histogram(h_in, h_out, bins, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("Histogram of %d elements with %d bins\n", arraySize, bins);
	if (checkResults(h_out, serialResult, bins))
		printf("Test Passed\n");
	else
		printf("Test Failed\n");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.


	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t histogram(int *h_in, int *h_out, unsigned int bins, unsigned int size)
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

	cudaStatus = cudaMalloc((void**)&d_out, bins * sizeof(int));
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

	cudaStatus = cudaMemcpy(d_out, h_out, bins * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	naive_histo_kernel<< <1, size>> >(d_out, d_in, bins);
	cudaStatus = cudaDeviceSynchronize();
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scan_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching scan_kernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(h_out, d_out, bins * sizeof(int), cudaMemcpyDeviceToHost);
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

void serialVersion(int* h_in, int* h_out, unsigned int bins, unsigned int size)
{
	for (int i = 0; i < bins; i++) 
	{
		h_out[i] = 0;
	}
	for (int i = 0; i < size; i++) 
	{
		h_out[COMPUTE_BIN(h_in[i])]++;
	}
}

bool checkResults(int* array1, int* array2, unsigned int size)
{
	bool result = true;
	for (unsigned int i = 0; i < size; i++)
	{
		if (array1[i] != array2[i])
		{
			result = false;
			return result;
		}
	}
	return result;
}