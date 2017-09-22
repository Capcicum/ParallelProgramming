//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
__global__ void simple_histo(unsigned int *d_bins, const int* const d_in, const unsigned int bitShift)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int myItem = d_in[myId];
	unsigned int bin = (myItem >> bitShift) & 1;
	atomicAdd(&(d_bins[bin]), 1);
}

__global__ void prescan(unsigned int *d_output, unsigned int *d_input, int size)
{
	extern __shared__ int temp[];  // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;

	temp[2 * thid] = d_input[2 * thid]; // load input into shared memory
	temp[2 * thid + 1] = d_input[2 * thid + 1];

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
	d_output[2 * thid] = temp[2 * thid]; // write results to device memory
	d_output[2 * thid + 1] = temp[2 * thid + 1];

}

__global__ void scatter(unsigned int *d_inputVals, const float* const d_in, float lumMin, const unsigned int bitShift, const size_t numElems)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int  myItem = d_inputVals[myId];

	if ((myItem >> bitShift) & 1)
	{

	}

}
for (unsigned int j = 0; j < numElems; ++j) {
	unsigned int bin = (vals_src[j] & mask) >> i;
	vals_dst[binScan[bin]] = vals_src[j];
	pos_dst[binScan[bin]] = pos_src[j];
	binScan[bin]++;
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE

	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks = numElems / maxThreadsPerBlock;

	int numOfBins = 2;
	unsigned int *d_bins;
	unsigned int *d_scan;
	unsigned int *d_inputV;
	unsigned int *d_inputP;

	checkCudaErrors(cudaMalloc(&d_bins, sizeof(unsigned int)*numOfBins));
	checkCudaErrors(cudaMalloc(&d_scan, sizeof(unsigned int)*numOfBins));
	checkCudaErrors(cudaMalloc(&d_inputV, sizeof(unsigned int)*numOfBins));
	checkCudaErrors(cudaMalloc(&d_inputP, sizeof(unsigned int)*numOfBins));

	checkCudaErrors(cudaMemcpy(d_inputV, d_outputVals, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_inputP, d_outputPos, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice));

	for (int i = 0; i < 8 * sizeof(unsigned int); i++);
	{
		checkCudaErrors(cudaMemset(*d_bins, 0, sizeof(unsigned int) * numOfBins));
		checkCudaErrors(cudaMemset(*d_scan, 0, sizeof(unsigned int) * numOfBins));

		unsigned int mask = i;
		simple_histo << <blocks, threads >> > (d_bins, d_inputV, mask);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		prescan << <blocks, threads >> >(unsigned int *d_scan, unsigned int *d_bins, numOfBins);



	}
}
