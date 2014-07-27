#include "stdafx.h"
#include "CUDA_Transfer.h"
using namespace cv;

extern "C" void run( IplImage *frame, PCA *decPCA, Mat pRecon, int threshold );

void run_cuda(IplImage *frame, PCA *decPCA, Mat pRecon, int threshold)
{
   run(frame, decPCA, pRecon, threshold);
   return;
}