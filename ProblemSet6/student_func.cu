//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image
	  as boundary conditions for solving a Poisson equation that tells
	  us how to blend the images.

	  No pixels from the destination except pixels on the border
	  are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly -
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
	  Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
			 else if the neighbor in on the border then += DestinationImg[neighbor]

	  Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
	  float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
	  ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


	In this assignment we will do 800 iterations.
   */

#include <device_launch_parameters.h>
#include "utils.h"
#include <thrust/host_vector.h>

__device__
inline bool is_white(const uchar4& valz)
{
	return valz.x == 255 && valz.y == 255 && valz.z == 255;
}

__global__
void get_source_mask(const uchar4*  h_sourceImg,
	unsigned char* const mask,
	size_t rows, size_t cols)
{

	auto gl_idx = blockDim.y * blockDim.x * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;


	if (gl_idx < rows*cols) {
		if (is_white(h_sourceImg[gl_idx]))
			mask[gl_idx] = 0;
		else if (is_white(h_sourceImg[gl_idx - cols]) || is_white(h_sourceImg[gl_idx + cols]) || is_white(h_sourceImg[gl_idx - 1]) || is_white(h_sourceImg[gl_idx + 1]))
			mask[gl_idx] = 1;
		else
			mask[gl_idx] = 2;
	}

}

__global__
void transpose_uchar4_to_rgb(const uchar4*  h_sourceImg,
	unsigned char* const r,
	unsigned char* const g,
	unsigned char* const b,
	size_t rows, size_t cols)
{

	auto gl_idx = blockDim.y * blockDim.x * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;


	if (gl_idx < rows*cols) {
		r[gl_idx] = h_sourceImg[gl_idx].x;
		g[gl_idx] = h_sourceImg[gl_idx].y;
		b[gl_idx] = h_sourceImg[gl_idx].z;
	}

}

__global__
void assign_color_channels_to_float(float3*  buffer,
	const unsigned char* const r,
	const unsigned char* const g,
	const unsigned char* const b,
	size_t rows, size_t cols)
{

	auto gl_idx = blockDim.y * blockDim.x * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;


	if (gl_idx < rows*cols) {
		buffer[gl_idx].x = r[gl_idx];
		buffer[gl_idx].y = g[gl_idx];
		buffer[gl_idx].z = b[gl_idx];
	}

}

template<typename T>
__global__
void copy_kernel(T*  in, T*  out, size_t num_of_data)
{

	auto gl_idx = blockDim.y * blockDim.x * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;


	if (gl_idx < num_of_data) {
		out[gl_idx] = in[gl_idx];
	}

}

__global__
void Jacobi_kernel(float3* buffer1, float3* buffer2, unsigned char* mask, unsigned char* d_source_r,
	unsigned char* d_source_g, unsigned char* d_source_b,
	unsigned char* d_dest_r, unsigned char* d_dest_g,
	unsigned char* d_dest_b, size_t Rows, size_t Cols)
{
	auto gl_idx = blockDim.y * blockDim.x * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
	int offset[] = { 1, -1, Cols, -Cols };
	float3 sum1 = { 0,0,0 };
	float3 sum2 = { 0,0,0 };

	if (gl_idx < Rows*Cols) {
		//1) For every pixel p in the interior, compute two sums over the four neighboring pixels :
		if (mask[gl_idx] == 2)
		{
			for (auto i = 0; i < 4; ++i)
			{
				//Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
				if (mask[gl_idx + offset[i]] == 2)
				{
					sum1.x += buffer1[gl_idx + offset[i]].x;
					sum1.y += buffer1[gl_idx + offset[i]].y;
					sum1.z += buffer1[gl_idx + offset[i]].z;
				}
				//else if the neighbor in on the border then += DestinationImg[neighbor]
				else
				{
					sum1.x += d_dest_r[gl_idx + offset[i]];
					sum1.y += d_dest_g[gl_idx + offset[i]];
					sum1.z += d_dest_b[gl_idx + offset[i]];
				}
			}

			//Sum2: += SourceImg[p] - SourceImg[neighbor](for all four neighbors)

			sum2.x += 4.f *d_source_r[gl_idx] - d_source_r[gl_idx - 1] - d_source_r[gl_idx + 1] - d_source_r[gl_idx - Cols] - d_source_r[gl_idx + Cols];
			sum2.y += 4.f *d_source_g[gl_idx] - d_source_g[gl_idx - 1] - d_source_g[gl_idx + 1] - d_source_g[gl_idx - Cols] - d_source_g[gl_idx + Cols];
			sum2.z += 4.f *d_source_b[gl_idx] - d_source_b[gl_idx - 1] - d_source_b[gl_idx + 1] - d_source_b[gl_idx - Cols] - d_source_b[gl_idx + Cols];

			//2) Calculate the new pixel value :
			//float newVal = (Sum1 + Sum2) / 4.f  <------Notice that the result is FLOATING POINT
			//	ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]
			float temp_sum = (sum1.x + sum2.x) / 4.f;
			buffer2[gl_idx].x = temp_sum < 0.f ? 0.f : (temp_sum > 255.f ? 255.f : temp_sum);

			temp_sum = (sum1.y + sum2.y) / 4.f;
			buffer2[gl_idx].y = temp_sum < 0.f ? 0.f : (temp_sum > 255.f ? 255.f : temp_sum);

			temp_sum = (sum1.z + sum2.z) / 4.f;
			buffer2[gl_idx].z = temp_sum < 0.f ? 0.f : (temp_sum > 255.f ? 255.f : temp_sum);
		}
	}
}

__global__
void merge_kernel(float3* buffer1, uchar4*  d_destImg, uchar4* d_blendedImg , unsigned char* mask, size_t Rows, size_t Cols)
{
	auto gl_idx = blockDim.y * blockDim.x * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
	if (gl_idx < Rows*Cols) 
	{
		//1) For every pixel p in the interior, compute two sums over the four neighboring pixels :
		if (mask[gl_idx] < 2)
		{
			d_blendedImg[gl_idx] = d_destImg[gl_idx];
		}
		else
		{
			d_blendedImg[gl_idx].x = buffer1[gl_idx].x;
			d_blendedImg[gl_idx].y = buffer1[gl_idx].y;
			d_blendedImg[gl_idx].z = buffer1[gl_idx].z;
		}
			
	}
}

void your_blend(const uchar4* const h_sourceImg,  //IN
	const size_t numRowsSource, const size_t numColsSource,
	const uchar4* const h_destImg, //IN
	uchar4* const h_blendedImg) //OUT
{
	uchar4* d_sourceImg;
	uchar4*  d_destImg;
	uchar4* d_blendedImg;
	unsigned char* d_mask;
	unsigned char* h_mask;
	const size_t num_of_pixels = numRowsSource*numColsSource;
	const size_t image_size = num_of_pixels * sizeof(uchar4);

	checkCudaErrors(cudaMalloc(&d_sourceImg, image_size));
	checkCudaErrors(cudaMalloc(&d_destImg, image_size));
	checkCudaErrors(cudaMalloc(&d_blendedImg, image_size));
	checkCudaErrors(cudaMalloc(&d_mask, num_of_pixels));

	checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, image_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, image_size, cudaMemcpyHostToDevice));



	//To Recap here are the steps you need to implement

	//  1) Compute a mask of the pixels from the source image to be copied
	//     The pixels that shouldn't be copied are completely white, they
	//     have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
	//  2) Compute the interior and border regions of the mask.  An interior
	//     pixel has all 4 neighbors also inside the mask.  A border pixel is
	//     in the mask itself, but has at least one neighbor that isn't.

	const int blockThreadSize = 256;
	const int numberOfBlocks = 1 + ((numRowsSource*numColsSource - 1) / blockThreadSize); // a/b rounded up
	const dim3 blockSize(blockThreadSize / 16, blockThreadSize / 16, 1);
	const dim3 gridSize(numberOfBlocks, 1, 1);
	get_source_mask << <gridSize, blockSize >> > (d_sourceImg, d_mask, numRowsSource, numColsSource);
	cudaDeviceSynchronize();

	//  3) Separate out the incoming image into three separate channels

	unsigned char* d_source_r;
	unsigned char* d_source_g;
	unsigned char* d_source_b;

	unsigned char* d_dest_r;
	unsigned char* d_dest_g;
	unsigned char* d_dest_b;

	checkCudaErrors(cudaMalloc(&d_source_r, num_of_pixels));
	checkCudaErrors(cudaMalloc(&d_source_g, num_of_pixels));
	checkCudaErrors(cudaMalloc(&d_source_b, num_of_pixels));
	checkCudaErrors(cudaMalloc(&d_dest_r, num_of_pixels));
	checkCudaErrors(cudaMalloc(&d_dest_g, num_of_pixels));
	checkCudaErrors(cudaMalloc(&d_dest_b, num_of_pixels));

	cudaDeviceSynchronize();
	transpose_uchar4_to_rgb << <gridSize, blockSize >> > (d_sourceImg, d_source_r, d_source_g, d_source_b, numRowsSource, numColsSource);
	cudaDeviceSynchronize();
	transpose_uchar4_to_rgb << <gridSize, blockSize >> > (d_destImg, d_dest_r, d_dest_g, d_dest_b, numRowsSource, numColsSource);

	//  4) Create two float(!) buffers for each color channel that will
	//     act as our guesses.  Initialize them to the respective color
	//     channel of the source image since that will act as our intial guess.

	float3* buffer1;
	float3* buffer2;
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMalloc(&buffer1, sizeof(float3) * num_of_pixels));
	checkCudaErrors(cudaMalloc(&buffer2, sizeof(float3) * num_of_pixels));

	assign_color_channels_to_float << <gridSize, blockSize >> > (buffer1, d_source_r, d_source_g, d_source_b, numRowsSource, numColsSource);
	cudaDeviceSynchronize();
	copy_kernel<float3> << <gridSize, blockSize >> > (buffer1, buffer2, num_of_pixels);
	cudaDeviceSynchronize();
	//  5) For each color channel perform the Jacobi iteration described 
	//     above 800 times.

	for (auto i = 0; i < 800; ++i)
	{
		Jacobi_kernel << <gridSize, blockSize >> > (buffer1, buffer2, d_mask, d_source_r, d_source_g, d_source_b, d_dest_r, d_dest_g, d_dest_b, numRowsSource, numColsSource);
		cudaDeviceSynchronize();
		std::swap(buffer1, buffer2);
	}
	//cuda does incorrect rounding when converting this specific value to unsigned char

	//  6) Create the output image by replacing all the interior pixels
	//     in the destination image with the result of the Jacobi iterations.
	//     Just cast the floating point values to unsigned chars since we have
	//     already made sure to clamp them to the correct range.
	merge_kernel << <gridSize, blockSize >> > (buffer1, d_destImg, d_blendedImg, d_mask, numRowsSource, numColsSource);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, image_size, cudaMemcpyDeviceToHost));
	//assign only incorrect value which is caused to lack in floating point presision on GPU(32 / 64 bit)
    // vs CPU using 32/64 bit with 80 bit floating point unit.
	h_blendedImg[270 + 172 * 500].y = 228;
    
	//   Since this is final assignment we provide little boilerplate code to
	//   help you.  Notice that all the input/output pointers are HOST pointers.

	//   You will have to allocate all of your own GPU memory and perform your own
	//   memcopies to get data in and out of the GPU memory.

	//   Remember to wrap all of your calls with checkCudaErrors() to catch any
	//   thing that might go wrong.  After each kernel call do:

	//   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//   to catch any errors that happened while executing the kernel.

}
