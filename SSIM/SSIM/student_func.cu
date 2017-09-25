
#include "utils.h"
#include <algorithm>

__global__ void filter_apply(float *d_mu, unsigned char*  d_in, float* d_filter, const size_t numElems)
{
	extern __shared__ float sdata[];

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int sold = blockDim.x;

	unsigned char myItem = d_in[myId];
	float myFilter = d_filter[6*6];

	*d_mu = myItem*myFilter;
	__syncthreads();            // make sure entire block is loaded!

								// do reduction in shared mem
	/*for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = sdata[tid] + sdata[tid + s];
		}
		if (tid == s - 1 && sold > 2 * s)
		{
			sdata[tid] = sdata[tid] + sdata[sold - 1];
		}
		__syncthreads();        // make sure all adds at one stage are done!
		sold = s;
	}*/

	// only thread 0 writes result for this block back to global mem
	/*if (tid == 0)
	{
		d_mu[blockIdx.x] = sdata[0];
	}*/
}

__global__ void shmem_reduce_kernel(float * d_out, const float* const d_in, bool type)
{
	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ float sdata[];

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int sold = blockDim.x;

	// load shared mem from global mem
	sdata[tid] = d_in[myId];
	__syncthreads();            // make sure entire block is loaded!

								// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			if (type)
				sdata[tid] = max(sdata[tid], sdata[tid + s]);
			else
				sdata[tid] = min(sdata[tid], sdata[tid + s]);
		}
		if (tid == s - 1 && sold > 2 * s)
		{
			if (type)
				sdata[tid] = max(sdata[tid], sdata[sold - 1]);
			else
				sdata[tid] = min(sdata[tid], sdata[sold - 1]);
		}
		__syncthreads();        // make sure all adds at one stage are done!
		sold = s;
	}

	// only thread 0 writes result for this block back to global mem
	if (tid == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}
}

void reduce(float * d_out, const float* const d_in, int size, bool type)
{
	// assumes that size is not greater than maxThreadsPerBlock^2
	// and that size is a multiple of maxThreadsPerBlock
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks = size / maxThreadsPerBlock;
	float *d_intermediate;

	checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(float)*size));

	shmem_reduce_kernel << <blocks, threads, threads * sizeof(float) >> >(d_intermediate, d_in, type);

	// now we're down to one block left, so reduce it
	// launch one thread for each block in prev step
	threads = blocks;
	blocks = 1;
	shmem_reduce_kernel << <blocks, threads, threads * sizeof(float) >> >(d_out, d_intermediate, type);

	checkCudaErrors(cudaFree(d_intermediate));
}

void your_gaussian_blur(const unsigned char* const h_inputNoiseGray, const unsigned char* const h_inputRefGray,
                        const unsigned int numRows, const unsigned int numCols,
                        float* d_filter, const int filterWidth)
{
  
	unsigned char* d_noise;
	unsigned char* d_ref;
	float* h_filter;
	float* d_mu;

	unsigned int numElems = numCols*numRows;
	unsigned int numFilter = filterWidth*filterWidth;

	unsigned int maxThreads = 1024;
	unsigned int maxBlocks = 1024;

	checkCudaErrors(cudaMalloc(&d_noise, sizeof(unsigned char)*numElems));
	checkCudaErrors(cudaMalloc(&d_ref, sizeof(unsigned char)*numElems));
	checkCudaErrors(cudaMalloc(&h_filter, sizeof(float)*numFilter));
	checkCudaErrors(cudaMalloc(&d_mu, sizeof(float)));


	checkCudaErrors(cudaMemcpy(d_noise, h_inputNoiseGray, numElems * sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ref, h_inputRefGray, numElems * sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(h_filter, d_filter, numFilter * sizeof(float), cudaMemcpyDeviceToHost));

	filter_apply << <1, 1 , sizeof(float)*121>> > (d_mu, d_ref, d_filter, numElems);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	float *h_mu = (float*)malloc(sizeof(float));

	checkCudaErrors(cudaMemcpy(h_mu, d_mu, sizeof(float), cudaMemcpyDeviceToHost));

	printf("%f\n", d_mu);



}



void cleanup() 
{

}


