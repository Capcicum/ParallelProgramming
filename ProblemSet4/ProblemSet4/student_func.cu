//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

#define makePow2(v) \
	v--;			\
	v |= v >> 1;	\
	v |= v >> 2;	\
	v |= v >> 4;	\
	v |= v >> 8;	\
	v |= v >> 16;	\
v++; \



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
__global__ void simple_histo(unsigned int *d_bins, unsigned int*  d_in, const unsigned int bitShift, const size_t numElems)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if (myId >= numElems)
		return;
	int myItem = d_in[myId];
	unsigned int bin = (myItem >> bitShift) & 1;
	atomicAdd(&(d_bins[bin]), 1);
}

__global__ void bit_position(unsigned int *d_ones, unsigned int *d_zeros, const unsigned int const *d_in, const unsigned int bitShift, const size_t numElems)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if (myId >= numElems)
		return;
	int myItem = d_in[myId];
	if ((myItem >> bitShift) & 1 == 1)
	{
		d_ones[myId] = 1;
		d_zeros[myId] = 0;
	}
	else
	{
		d_ones[myId] = 0;
		d_zeros[myId] = 1;
	}
}

//Taken from: https://github.com/kadircet/cs344/blob/master/ps4/student_func.cu
__global__ void scan(const unsigned int* const d_in, unsigned int* const d_out, unsigned int* const d_blockout, int n)
{
	extern __shared__ unsigned int l_mod[];

	int bid = blockIdx.x + gridDim.x*blockIdx.y + gridDim.x*gridDim.y*blockIdx.z;
	int id = bid*blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	unsigned int tmp;

	if (id >= n || bid>n / blockDim.x)
		return;
	l_mod[tid] = d_in[id];
	__syncthreads();

	for (unsigned int s = 1; s<blockDim.x; s <<= 1)
	{
		tmp = l_mod[tid];
		__syncthreads();

		if (tid + s<blockDim.x)
			l_mod[tid + s] += tmp;
		__syncthreads();
	}
	d_out[id] = tid>0 ? l_mod[tid - 1] : 0;
	if (tid == 1023)
		d_blockout[bid] = l_mod[tid];
}

//Taken from: https://github.com/kadircet/cs344/blob/master/ps4/student_func.cu
__global__ void add(const unsigned int* const d_in, unsigned int* const d_out, const unsigned int* d_incr, int n)
{
	int bid = blockIdx.x + gridDim.x*blockIdx.y + gridDim.x*gridDim.y*blockIdx.z;
	int id = bid*blockDim.x + threadIdx.x;
	if (id >= n)
		return;
	d_out[id] = d_in[id] + d_incr[bid];
}

//Taken from: https://github.com/kadircet/cs344/blob/master/ps4/student_func.cu
void prefixSum(unsigned int* d_arr, unsigned int* d_out, int size)
{
	unsigned int* d_blockout;
	int n = size;
	makePow2(n);
	int threads = 1024;
	dim3 blocks(n / threads, n / threads / 65535 + 1, n / threads / 65535 / 65535 + 1);
	blocks.x = min(blocks.x, 65535);
	blocks.y = min(blocks.y, 65535);
	blocks.z = min(blocks.z, 65535);
	blocks.x = max(1, blocks.x);
	checkCudaErrors(cudaMalloc(&d_blockout, sizeof(unsigned int)*blocks.x*blocks.y*blocks.z));
	scan << <blocks, threads, threads * sizeof(unsigned int) >> >(d_arr, d_out, d_blockout, size);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	if (blocks.x>1)
	{
		prefixSum(d_blockout, d_blockout, blocks.x*blocks.y*blocks.z);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		add << <blocks, threads >> >(d_out, d_out, d_blockout, size);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
	cudaFree(d_blockout);
}


__global__ void sort(unsigned int* const d_inputVals, unsigned int* const d_inputPos,
	unsigned int* const d_outputVals, unsigned int* const d_outputPos,
	unsigned int* d_ones_scanned, unsigned int* d_zeros_scanned,
	unsigned int* d_bins, const unsigned int bitShift, const size_t numElems)
{
	int thid = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (thid >= numElems)
		return;

	unsigned int bit = (d_inputVals[thid] >> bitShift) & 1;

	int outp = 0;

	if (bit == 1)
		outp = d_bins[1] + d_ones_scanned[thid];
	else
		outp = d_bins[0] + d_zeros_scanned[thid];

	d_outputVals[outp] = d_inputVals[thid];  
	d_outputPos[outp] = d_inputPos[thid];
}

__global__ void swap(unsigned int* const d_in, unsigned int* d_out, const size_t numElems)
{
	int thid = threadIdx.x + blockDim.x * blockIdx.x;
	if (thid >= numElems)
		return;
	int myItem = d_in[thid];
	d_in[thid] = d_out[thid];
	d_out[thid] = myItem;
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
	int blocks = numElems / maxThreadsPerBlock + 1;

	//printf("blocks: %d, numElems: %d numThreads: %d\n", blocks, numElems, blocks*threads);

	int numOfBins = 2;
	unsigned int *d_bins;
	unsigned int *d_scan;
	unsigned int *d_zeros;
	unsigned int *d_ones;
	unsigned int *d_ones_scanned;
	unsigned int *d_zeros_scanned;

	/*unsigned int *h_hist;
	unsigned int *h_scan;
	unsigned int *h_zeros;
	unsigned int *h_ones;
	unsigned int *h_ones_scanned;
	unsigned int *h_zeros_scanned;

	unsigned int *h_inputVals;
	unsigned int* h_inputPos;
	unsigned int* h_outputVals;
	unsigned int* h_outputPos;

	h_inputVals = (unsigned int*)malloc(sizeof(unsigned int)*numElems);
	h_inputPos = (unsigned int*)malloc(sizeof(unsigned int)*numElems);
	h_outputVals = (unsigned int*)malloc(sizeof(unsigned int)*numElems);
	h_outputPos = (unsigned int*)malloc(sizeof(unsigned int)*numElems);

	h_hist = (unsigned int*) malloc(sizeof(unsigned int)*numOfBins);
	h_scan = (unsigned int*)malloc(sizeof(unsigned int)*numOfBins);
	h_zeros = (unsigned int*)malloc(sizeof(unsigned int)*numElems);
	h_ones = (unsigned int*)malloc(sizeof(unsigned int)*numElems);
	h_ones_scanned = (unsigned int*)malloc(sizeof(unsigned int)*numElems);
	h_zeros_scanned = (unsigned int*)malloc(sizeof(unsigned int)*numElems);*/


	checkCudaErrors(cudaMalloc(&d_bins, sizeof(unsigned int)*numOfBins));
	checkCudaErrors(cudaMalloc(&d_scan, sizeof(unsigned int)*numOfBins));
	checkCudaErrors(cudaMalloc(&d_zeros, sizeof(unsigned int)*numElems));
	checkCudaErrors(cudaMalloc(&d_ones, sizeof(unsigned int)*numElems));
	checkCudaErrors(cudaMalloc(&d_ones_scanned, sizeof(unsigned int)*numElems));
	checkCudaErrors(cudaMalloc(&d_zeros_scanned, sizeof(unsigned int)*numElems));

	/*cudaMemcpy(h_inputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	unsigned int *binHistogram = new unsigned int[numOfBins];
	unsigned int *binScan = new unsigned int[numOfBins];
	unsigned int *ones = new unsigned int[numElems];
	unsigned int *zeros = new unsigned int[numElems];

	unsigned int *ones_scanned = new unsigned int[numElems];
	unsigned int *zeros_scanned = new unsigned int[numElems];

	unsigned int *vals_dst = new unsigned int[numElems];
	unsigned int *pos_dst = new unsigned int[numElems];

	memset(ones_scanned, 0, sizeof(unsigned int) * numElems);
	memset(zeros_scanned, 0, sizeof(unsigned int) * numElems);*/

	for (int i = 0; i < 8 * (int)sizeof(unsigned int); i++)
	{

		//printf("Iteration: %d\n", i);
		checkCudaErrors(cudaMemset(d_bins, 0, sizeof(unsigned int) * numOfBins));
		checkCudaErrors(cudaMemset(d_scan, 0, sizeof(unsigned int) * numOfBins));

		/*memset(binHistogram, 0, sizeof(unsigned int) * numOfBins);
		memset(binScan, 0, sizeof(unsigned int) * numOfBins);*/

		simple_histo << <blocks, threads>> > (d_bins, d_inputVals, i, numElems);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		/*cudaMemcpy(h_hist, d_bins, numOfBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);

		for (unsigned int j = 0; j < numElems; ++j) {
			unsigned int bin = (h_inputVals[j] >> i) & 1;
			binHistogram[bin]++;
		}

		for (unsigned int j = 0; j < numOfBins; ++j)
		{
			if (binHistogram[j] != h_hist[j])
			{
				printf("Histogram error at pos %d should be: %d is: %d", j, binHistogram[j], h_hist[j]);
				return;
			}
		}*/

		prefixSum(d_bins, d_scan, numOfBins);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		/*cudaMemcpy(h_scan, d_scan, numOfBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);

		for (unsigned int j = 1; j < numOfBins; ++j) {
			binScan[j] = binScan[j - 1] + binHistogram[j - 1];
		}

		for (unsigned int j = 0; j < numOfBins; ++j)
		{
			if (binScan[j] != h_scan[j])
			{
				printf("Histogram Scan error at pos %d should be: %d is: %d", j, binScan[j], h_scan[j]);
				return;
			}
		}*/
		
		bit_position << < blocks, threads >> > (d_ones, d_zeros, d_inputVals, i, numElems);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		/*cudaMemcpy(h_ones, d_ones, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_zeros, d_zeros, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);

		for (unsigned int j = 0; j < numElems; j++)
		{
			if ((h_inputVals[j] >> i) & 1)
			{
				ones[j] = 1;
				zeros[j] = 0;
			}
			else
			{
				ones[j] = 0;
				zeros[j] = 1;
			}
		}

		for (unsigned int j = 0; j < numElems; ++j)
		{
			if (h_ones[j] != ones[j])
			{
				printf("Bit position ones error at pos %d should be: %d is: %d", j, ones[j], h_ones[j]);
				return;
			}
			if (h_zeros[j] != zeros[j])
			{
				printf("Bit position zeros error at pos %d should be: %d is: %d", j, zeros[j], h_zeros[j]);
				return;
			}
		}*/

		prefixSum(d_ones, d_ones_scanned, numElems);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		/*cudaMemcpy(h_ones_scanned, d_ones_scanned, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);

		for (unsigned int j = 1; j < numElems; ++j) {
			ones_scanned[j] = ones_scanned[j - 1] + ones[j - 1];
		}

		for (unsigned int j = 0; j < numElems; ++j)
		{
			if (ones_scanned[j] != h_ones_scanned[j])
			{
				printf("Histogram Scan error at pos %d should be: %d is: %d", j, ones_scanned[j], h_ones_scanned[j]);
				return;
			}
		}*/

		prefixSum(d_zeros, d_zeros_scanned, numElems);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		/*cudaMemcpy(h_zeros_scanned, d_zeros_scanned, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);

		for (unsigned int j = 1; j < numElems; ++j) {
			zeros_scanned[j] = zeros_scanned[j - 1] + zeros[j - 1];
		}

		for (unsigned int j = 0; j < numElems; ++j)
		{
			if (zeros_scanned[j] != h_zeros_scanned[j])
			{
				printf("Histogram Scan error at pos %d should be: %d is: %d", j, zeros_scanned[j], h_zeros_scanned[j]);
				return;
			}
		}*/

		sort << < blocks, threads >> > (d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_ones_scanned, d_zeros_scanned, d_scan, i, numElems);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		/*cudaMemcpy(h_inputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_inputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_outputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_outputPos, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);

		for (unsigned int j = 0; j < numElems; ++j) {
			unsigned int bin = (h_inputVals[j] >> i) & 1;
			vals_dst[binScan[bin]] = h_inputVals[j];
			pos_dst[binScan[bin]] = h_inputPos[j];
			binScan[bin]++;
		}

		for (unsigned int j = 0; j < numElems; ++j)
		{
			if (vals_dst[j] == h_outputVals[j] && pos_dst[j] == h_outputPos[j])
			{

			}
			else
			{
				printf("Sort vals error at pos %d should be: %d is: %d\n", j, vals_dst[j], h_outputVals[j]);
				printf("Sort pos error at pos %d should be: %d is: %d\n", j, pos_dst[j], h_outputPos[j]);
				return;
			}
		}*/

		swap << < blocks, threads >> >(d_inputPos, d_outputPos, numElems);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		swap << < blocks, threads >> >(d_inputVals, d_outputVals, numElems);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		/*cudaMemcpy(h_inputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_inputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_outputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_outputPos, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);*/
	}
	cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
