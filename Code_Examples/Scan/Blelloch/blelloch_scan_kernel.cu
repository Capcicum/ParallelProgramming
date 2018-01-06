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

cudaError_t scan(int *h_in, int *h_out, unsigned int size);
void createArray(const int size, int* array);
void serialVersion(int* h_in, int* h_out, unsigned int size);
bool checkResults(int* array1, int* array2, unsigned int size);


// Blelloch  scan 
__global__ void scan_kernel(int *d_out, int *d_in, int size)
{
	extern __shared__ int temp[];  // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;

	temp[2 * thid] = d_in[2 * thid]; // load input into shared memory
	temp[2 * thid + 1] = d_in[2 * thid + 1];
	__syncthreads();
	for (int d = size >> 1; d > 0; d >>= 1)                    // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0) { temp[size - 1] = 0; } // clear the last element
	__syncthreads();

	for (int d = 1; d < size; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{

			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	d_out[2 * thid] = temp[2 * thid]; // write results to device memory
	d_out[2 * thid + 1] = temp[2 * thid + 1];
}

int main()
{
	const int arraySize = 256;
	int h_in[arraySize] = { 0 };
	int h_out[arraySize] = { 0 };
	int serialResult[arraySize] = { 0 };

	createArray(arraySize, h_in);

	// Scan in serial.
	serialVersion(h_in, serialResult, arraySize);

	// Scan in parallel.
	cudaError_t cudaStatus = scan(h_in, h_out, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("Scan of %d elements\n", arraySize);
	if (checkResults(h_out, serialResult, arraySize))
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
	int lol;
	std::cin >> lol;
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t scan(int *h_in, int *h_out, unsigned int size)
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

	cudaStatus = cudaMalloc((void**)&d_out, size * sizeof(int));
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

	cudaStatus = cudaMemcpy(d_out, h_out, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	scan_kernel << <1, size/2, size * sizeof(int) >> >(d_out, d_in, size);
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
	cudaStatus = cudaMemcpy(h_out, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);
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

void serialVersion(int* h_in, int* h_out, unsigned int size)
{
	h_out[0] = 0; // since this is a prescan, not a scan
	for (unsigned int j = 1; j < size; ++j)
	{
		h_out[j] = h_in[j - 1] + h_out[j - 1];
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