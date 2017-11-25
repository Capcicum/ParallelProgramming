//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>
#include <assert.h>
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

/* TODO: ALOT COULD BE OPTIMIZED 
         FOR INSTANCE USING SHARED MEMORY IN KERNELS 
         ALOT OF COPIES AND MEMSET ARE USED
         MAYBE COMBINE SOME OF THE KERNELS (FOR INSTANCE ZEROS AND ONES)
   REMEMBER THE DEBUG FLAG!!!!
*/
__global__
void histogram_kernel(unsigned int* const d_in, const size_t d_in_size, unsigned int* d_bins, const unsigned int bit_nbr )
{  
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if( myId >= d_in_size ) return;
    /* if the bit is set, increment the bin holding the number of ones (d_bins[1])
       otherwise increment the bin holder the number of zeros (d_bins[0])
       Must be done atomically as the bin is incremented by each thread */
    if( ( d_in[myId] & (1 << bit_nbr) ) != 0 ) atomicAdd( &d_bins[1], 1 );
    else atomicAdd( &d_bins[0], 1 );
}

__global__ 
void in_seg_block_scan_kernel(unsigned int* d_in, unsigned int* d_out, int size, unsigned int* d_incr )
{
    /* perform Hillis/Steele "segmented" (each block) inclusive scan
       The scan is converted to exclusive at the end and the "last" element
       that is not included in the exclusive scan is stored in "d_incr"
    */
	int bid = blockIdx.x; // block id
	int id = threadIdx.x + blockDim.x * bid; // id among all threads across all blocks
	int tid = threadIdx.x; // thread id in a block

    if( id >= size ) return;

    /* As it is a scan per block, is the size id larger than a block, the size should be the blocksize */
    unsigned int act_siz = min(blockDim.x, size);
    /* perform the actual hillis steele scan.
       remember that the result should be stored in the highest tid element number! */
    for( unsigned int s = 1; s < act_siz; s *= 2)
    {              
        if( tid + s < act_siz )
        {
            d_in[id+s] += d_in[id]; 
        }
        __syncthreads(); // per block synchronization!!!
    }
    /* the last thread in a block should store its value in d_incr
       as it is removed in the conversion to an exclusive scan */
    if ( tid == (blockDim.x - 1) && (d_incr != NULL) )
    {
        d_incr[bid] = d_in[id];
    }
    /* convert from inclusive scan to exclusive scan */
    d_out[id] = (tid > 0) ? d_in[id-1] : 0;
}

__global__ 
void add_incr_kernel(unsigned int* d_in, const size_t size, unsigned int* d_incr )
// must be performed after "in_seg_block_scan_kernel if blockDim.x > 1
{
    unsigned int bid = blockIdx.x;
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= size ) return;
    /* adds the value in d_incr to all elements related to a block
       this can be used to combine the segmented scans */
    d_in[id] += d_incr[bid];       
}

__global__
void predicate_kernel( unsigned int* const d_in, unsigned int* const d_out,
                       size_t size, bool pred, unsigned int bit_nbr )
// pred == true --> ones, pred == false --> zeros
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= size ) return;
    /* true if the bit is set, otherwise false */
    bool set = ( d_in[id] & (1 << bit_nbr) ) != 0;
    /* set all ones to 1 or all zeros to 1, depending on "pred" */
    d_out[id] = pred ? (set ? 1 : 0) : (set ? 0 : 1);
}

__global__
void move_kernel( unsigned int* const d_pred_zeros, unsigned int* const d_index_zeros,
                  unsigned int* const d_pred_ones, unsigned int* const d_index_ones,
                  unsigned int* const d_in_val, unsigned int* const d_in_pos,
                  unsigned int* const d_out_val, unsigned int* const d_out_pos,
                  unsigned int* const d_bin, const size_t size )
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= size ) return;

    /* if the predicate is set, get scatter address from offset in d_bin and index in d_index_
       then scatter the input value to ouput value, and the same for positions */
    if( d_pred_zeros[id] == 1 )
    {
      d_out_val[d_bin[0]+d_index_zeros[id]] = d_in_val[id];
      d_out_pos[d_bin[0]+d_index_zeros[id]] = d_in_pos[id]; 
    }
    if( d_pred_ones[id] == 1 )
    {
      d_out_val[d_bin[1]+d_index_ones[id]] = d_in_val[id];
      d_out_pos[d_bin[1]+d_index_ones[id]] = d_in_pos[id]; 
    }
}

__global__ void swap(unsigned int* const d_in, unsigned int* d_out, const size_t numElems)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numElems) return;
    /* swaps input and output
       can be used to ping pong values in a radix sort */
	unsigned int tmp = d_in[id];
	d_in[id] = d_out[id];
	d_out[id] = tmp;
}

#define DEBUG 1 // if "1" then serial validation routines are executed and prints are used
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
    const size_t nbr_bins = 2;
    const size_t nbr_bins_size = nbr_bins * sizeof( unsigned int );
    const size_t numElems_size = numElems * sizeof( unsigned int );

    unsigned int* d_bins; // contains a histogram of ones and zeroes
    unsigned int* d_bins_scan; // contains a scanned histogram of ones and zeroes
    unsigned int* d_pred_ones_old; // holds predicates for 1's
    unsigned int* d_pred_zeros_old; // holds predicates for 1's
    unsigned int* d_pred_ones; // holds predicates for 1's (being changed later, therefore the old value)
    unsigned int* d_pred_zeros; // holds predicates for 1's (being changed later, therefore the old value)
    unsigned int* d_index_ones; // holds index for scattering after indexing
    unsigned int* d_index_zeros; // holds index for scattering after indexing
    unsigned int* d_incr; // used for holding the last element in a block scan (to be added to next block)
    unsigned int* d_incr_scan; // scanned incr

#if DEBUG
    /* memory to hold device data to be able to print it */
    unsigned int* h_inputVals = (unsigned int*)malloc(sizeof(unsigned int)*numElems);
    unsigned int* h_hist = (unsigned int*)malloc(sizeof(unsigned int)*nbr_bins);
    unsigned int* h_scan = (unsigned int*)malloc(sizeof(unsigned int)*nbr_bins);
    unsigned int* h_zeros = (unsigned int*)malloc(sizeof(unsigned int)*numElems);
	unsigned int* h_ones = (unsigned int*)malloc(sizeof(unsigned int)*numElems);
    unsigned int* h_ones_scanned = (unsigned int*)malloc(sizeof(unsigned int)*numElems);
	unsigned int* h_zeros_scanned = (unsigned int*)malloc(sizeof(unsigned int)*numElems);
    /* memory to serially create data to compare to the parallel implementation */
    unsigned int* binHistogram = new unsigned int[nbr_bins];
    unsigned int* binScan = new unsigned int[nbr_bins];
    unsigned int* ones = new unsigned int[numElems];
	unsigned int* zeros = new unsigned int[numElems];
    unsigned int* ones_scanned = new unsigned int[numElems];
	unsigned int* zeros_scanned = new unsigned int[numElems];
#endif
    const dim3 thread_dim( 1024 );
    const dim3 grid_dim( ceil( static_cast<float>(numElems) / static_cast<float>(thread_dim.x) ) );
#if DEBUG
    printf("blocks: %u, threads: %u\n", grid_dim.x, thread_dim.x);
#endif
    checkCudaErrors( cudaMalloc( &d_bins, nbr_bins_size ) );
    checkCudaErrors( cudaMalloc( &d_bins_scan, nbr_bins_size ) );
    checkCudaErrors( cudaMalloc( &d_pred_ones_old, numElems_size ) );
    checkCudaErrors( cudaMalloc( &d_pred_zeros_old, numElems_size ) );
    checkCudaErrors( cudaMalloc( &d_pred_ones, numElems_size ) );
    checkCudaErrors( cudaMalloc( &d_pred_zeros, numElems_size ) );
    checkCudaErrors( cudaMalloc( &d_index_ones, numElems_size) );
    checkCudaErrors( cudaMalloc( &d_index_zeros, numElems_size) );
    checkCudaErrors( cudaMalloc( &d_incr, grid_dim.x * sizeof(unsigned int)) );
    checkCudaErrors( cudaMalloc( &d_incr_scan, grid_dim.x * sizeof(unsigned int)) );

#if DEBUG
    printf("Number of elements: %li\n", numElems);
#endif
    /* for loop for all bins the the input values (32 bit in this case) */
    for( unsigned int bit = 0; bit < sizeof( unsigned int ) * 8; bit++ )
    {   
        /* must be cleared each time as the are being incremented */     
        checkCudaErrors( cudaMemset( d_bins, 0, nbr_bins_size ) );
        checkCudaErrors( cudaMemset( d_bins_scan, 0, nbr_bins_size ) );

#if DEBUG
        checkCudaErrors( cudaMemcpy(h_inputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
        printf("bit: %u\n", bit);
#endif
    /* 1) Histogram of the number of occurrences of each digit */
        histogram_kernel<<< grid_dim, thread_dim >>>( d_inputVals, numElems, d_bins, bit );     
        cudaDeviceSynchronize(); checkCudaErrors( cudaGetLastError() );
#if DEBUG
        /* reference test */
        memset(binHistogram, 0, sizeof(unsigned int) * nbr_bins);
		checkCudaErrors( cudaMemcpy(h_hist, d_bins, nbr_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		for (unsigned int j = 0; j < numElems; ++j) {
			unsigned int bin = (h_inputVals[j] >> bit) & 1;
			binHistogram[bin]++;
		}
		for (unsigned int j = 0; j < nbr_bins; ++j)
		{
			if (binHistogram[j] != h_hist[j])
			{
				printf("Histogram error at pos %d should be: %d is: %d", j, binHistogram[j], h_hist[j]);
				return;
			}
		}
#endif
    /* 2) Exclusive Prefix Sum of Histogram */
        in_seg_block_scan_kernel<<< grid_dim, thread_dim >>>(d_bins, d_bins_scan, nbr_bins, d_incr );
        cudaDeviceSynchronize(); checkCudaErrors( cudaGetLastError() );
#if DEBUG
        /* reference test */
        memset(binScan, 0, sizeof(unsigned int) * nbr_bins);
		checkCudaErrors( cudaMemcpy(h_scan, d_bins_scan, nbr_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
        printf("zeros index: %u, ones index: %u\n", h_scan[0], h_scan[1]);
		for (unsigned int j = 1; j < nbr_bins; ++j) {
			binScan[j] = binScan[j - 1] + binHistogram[j - 1];
		}
		for (unsigned int j = 0; j < nbr_bins; ++j)
		{
			if (binScan[j] != h_scan[j])
			{
				printf("Histogram Scan error at pos %d should be: %d is: %d", j, binScan[j], h_scan[j]);
				return;
			}
		}
#endif
    /* 3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2] */

        /* create predicates for zeros and ones */
        predicate_kernel<<< grid_dim, thread_dim >>>( d_inputVals, d_pred_zeros, numElems, false, bit);
        cudaDeviceSynchronize(); checkCudaErrors( cudaGetLastError() );
        checkCudaErrors( cudaMemcpy( d_pred_zeros_old, d_pred_zeros, numElems_size, cudaMemcpyDeviceToDevice ) );
        predicate_kernel<<< grid_dim, thread_dim >>>( d_inputVals, d_pred_ones, numElems, true, bit);
        cudaDeviceSynchronize(); checkCudaErrors( cudaGetLastError() );
        checkCudaErrors( cudaMemcpy( d_pred_ones_old, d_pred_ones, numElems_size, cudaMemcpyDeviceToDevice ) );
#if DEBUG
        /* reference test */
		checkCudaErrors( cudaMemcpy(h_ones, d_pred_ones_old, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(h_zeros, d_pred_zeros_old, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		for (unsigned int j = 0; j < numElems; j++)
		{
			if ((h_inputVals[j] >> bit) & 1)
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
		}
#endif
        /* make segmented (block level) exclusive scan of predicate */
        in_seg_block_scan_kernel<<< grid_dim, thread_dim >>>(d_pred_zeros, d_index_zeros, numElems, d_incr );
        cudaDeviceSynchronize(); checkCudaErrors( cudaGetLastError() );
        /* make exclusive scan of the values scored in d_incr
           it holds the last values from each block before converting from inclusive to exclusive scan
           a scan is necessary as the values to be added must be accumulated to combine the segmented scans */
        in_seg_block_scan_kernel<<< grid_dim, thread_dim >>>(d_incr, d_incr_scan, grid_dim.x, NULL );
        cudaDeviceSynchronize(); checkCudaErrors( cudaGetLastError() );
        /* add the scanned values to the indexes to generate the actual indexes! */
        add_incr_kernel<<< grid_dim, thread_dim >>>( d_index_zeros, numElems, d_incr_scan );
        cudaDeviceSynchronize(); checkCudaErrors( cudaGetLastError() );;
#if DEBUG
		checkCudaErrors( cudaMemcpy(h_zeros_scanned, d_index_zeros, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
        memset(zeros_scanned, 0, sizeof(unsigned int) * numElems);
		for (unsigned int j = 1; j < numElems; ++j)
        {
			zeros_scanned[j] = zeros_scanned[j - 1] + zeros[j - 1];
		}
		for (unsigned int j = 0; j < numElems; ++j)
		{
			if (zeros_scanned[j] != h_zeros_scanned[j])
			{
				printf("Index zeros scan error at pos %d should be: %d is: %d", j, zeros_scanned[j], h_zeros_scanned[j]);
				return;
			}
		}
#endif
        /* same as the above description, but for ones instead of zeros */
        in_seg_block_scan_kernel<<< grid_dim, thread_dim >>>(d_pred_ones, d_index_ones, numElems, d_incr );
        cudaDeviceSynchronize(); checkCudaErrors( cudaGetLastError() );
        in_seg_block_scan_kernel<<< grid_dim, thread_dim >>>(d_incr, d_incr_scan, grid_dim.x, NULL );
        cudaDeviceSynchronize(); checkCudaErrors( cudaGetLastError() );
        add_incr_kernel<<< grid_dim, thread_dim >>>( d_index_ones, numElems, d_incr_scan );
        cudaDeviceSynchronize(); checkCudaErrors( cudaGetLastError() );
#if DEBUG
		checkCudaErrors( cudaMemcpy(h_ones_scanned, d_index_ones, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
        memset(ones_scanned, 0, sizeof(unsigned int) * numElems);
		for (unsigned int j = 1; j < numElems; ++j) {
			ones_scanned[j] = ones_scanned[j - 1] + ones[j - 1];
		}
		for (unsigned int j = 0; j < numElems; ++j)
		{
			if (ones_scanned[j] != h_ones_scanned[j])
			{
				printf("Index ones scan error at pos %d should be: %d is: %d", j, ones_scanned[j], h_ones_scanned[j]);
				return;
			}
		}
#endif  
    /* 4) Combine the results of steps 2 & 3 to determine the final
          output location for each element and move it there */
        /* scatter the input values to the "calculated" indexes (doing the actual sort for one bit) */
        move_kernel<<< grid_dim, thread_dim >>>
                     ( d_pred_zeros_old, d_index_zeros,
                       d_pred_ones_old,  d_index_ones,
                       d_inputVals,  d_inputPos,
                       d_outputVals, d_outputPos,
                       d_bins_scan, numElems );
        cudaDeviceSynchronize(); checkCudaErrors( cudaGetLastError() );

        /* swap input and output */
		swap <<< grid_dim, thread_dim >>>(d_inputPos, d_outputPos, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		swap <<< grid_dim, thread_dim >>>(d_inputVals, d_outputVals, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());        
    }

    /* make it fail! */
    // checkCudaErrors( cudaMemset( d_outputVals, 9642898, 1000*sizeof(unsigned int) ) );

    /* ensure that the sorted values end up in the output memory */
    checkCudaErrors( cudaMemcpy( d_outputVals, d_inputVals, numElems_size, cudaMemcpyDeviceToDevice ) ); 
    checkCudaErrors( cudaMemcpy( d_outputPos, d_inputPos, numElems_size, cudaMemcpyDeviceToDevice ) ); 

    /* remember to free the allocated memory */
    checkCudaErrors( cudaFree( d_bins ) );
    checkCudaErrors( cudaFree( d_bins_scan ) );
    checkCudaErrors( cudaFree( d_pred_ones_old ) );
    checkCudaErrors( cudaFree( d_pred_zeros_old ) );
    checkCudaErrors( cudaFree( d_pred_ones ) );
    checkCudaErrors( cudaFree( d_pred_zeros ) );
    checkCudaErrors( cudaFree( d_index_ones ) );
    checkCudaErrors( cudaFree( d_index_zeros ) );
    checkCudaErrors( cudaFree( d_incr ) );
    checkCudaErrors( cudaFree( d_incr_scan ) );
}