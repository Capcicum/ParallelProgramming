/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

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
	
	shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in, type);

	// now we're down to one block left, so reduce it
	// launch one thread for each block in prev step
	threads = blocks; 
	blocks = 1;
	shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate, type);

	checkCudaErrors(cudaFree(d_intermediate));
}

__global__ void simple_histo(unsigned int *d_bins, const float* const d_in, float lumMin, float lumRange, const int numBins)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	float myItem = d_in[myId];
	int myBin = (myItem - lumMin) / lumRange * numBins;
	atomicAdd(&(d_bins[myBin]), 1);
}


//https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

__global__ void prescan(unsigned int *g_odata, unsigned int *g_idata, int n)
{
	extern __shared__ int temp[];  // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;

	temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
	temp[2 * thid + 1] = g_idata[2 * thid + 1];

	for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree
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

	if (thid == 0) { temp[n - 1] = 0; } // clear the last element

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
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
	g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
	g_odata[2 * thid + 1] = temp[2 * thid + 1];

}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	float *d_min, *d_max;
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks = (numCols*numRows)/maxThreadsPerBlock;
	unsigned int *d_bins;

	checkCudaErrors(cudaMalloc(&d_min, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_bins, sizeof(unsigned int)*numBins));

	reduce(d_max, d_logLuminance, numRows*numCols, 1);
	reduce(d_min, d_logLuminance, numRows*numCols, 0);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost);

	float lumRange = max_logLum - min_logLum;
	
	simple_histo<<<blocks, threads>>>(d_bins, d_logLuminance, min_logLum, lumRange, numBins);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	prescan<<<1, numBins/2, threads * sizeof(unsigned int)>>>(d_cdf, d_bins, numBins);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaFree(d_min));
	checkCudaErrors(cudaFree(d_max));
	checkCudaErrors(cudaFree(d_bins));

}
