//Udacity HW2 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

//include the definitions of the above functions for this homework
#include "HW2.cpp"


/*******  DEFINED IN student_func.cu *********/

void your_gaussian_blur(const unsigned char* const h_inputNoiseGray, const unsigned char* const h_inputRefGray,
	const unsigned int numRows, const unsigned int numCols,
	float* filter, const int filterWidth);


/*******  Begin main *********/

int main(int argc, char **argv) {
  unsigned char *h_refImageGray,  *h_noiseImageGray;

  float *d_filter;
  unsigned int filterWidth;
  unsigned int numCols;
  unsigned int numRows;

  std::string ref_input_file;
  std::string noise_input_file;

  ref_input_file = "cinque_terre_small.jpg";
  noise_input_file = "cinque_terre_small_blur.png";

  double perPixelError = 0.0;
  double globalError   = 0.0;
  bool useEpsCheck = false;

  /*switch (argc)
  {
	case 2:
		ref_input_file = std::string(argv[1]);
		noise_input_file = "HW2_output.png";
	  break;
	case 3:
	  input_file  = std::string(argv[1]);
      output_file = std::string(argv[2]);
	  reference_file = "HW2_reference.png";
	  break;
	case 4:
	  input_file  = std::string(argv[1]);
      output_file = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  break;
	case 6:
	  useEpsCheck=true;
	  input_file  = std::string(argv[1]);
	  output_file = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  perPixelError = atof(argv[4]);
      globalError   = atof(argv[5]);
	  break;
	default:
      std::cerr << "Usage: ./HW2 input_file [output_filename] [reference_filename] [perPixelError] [globalError]" << std::endl;
      exit(1);
  }*/
  //load the image and give us our input and output pointers
  preProcess(&h_refImageGray, &h_noiseImageGray, &d_filter, &filterWidth, &numCols, &numRows, ref_input_file, noise_input_file);

  //allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);
  GpuTimer timer;

  timer.Start();

  //call the students' code
  your_gaussian_blur(h_refImageGray, h_noiseImageGray, numRows, numCols, d_filter, filterWidth);

  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //check results and output the blurred image

  /*size_t numPixels = numRows()*numCols();

  checkCudaErrors(cudaFree(d_redBlurred));
  checkCudaErrors(cudaFree(d_greenBlurred));
  checkCudaErrors(cudaFree(d_blueBlurred));

  cleanUp();*/

  return 0;
}
