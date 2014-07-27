#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "math.h"

using namespace cv;

int run_cuda(IplImage *frame, Mat eigenVector, Mat eigenValue, Mat mean);
int run_cuda_initialization( Mat Image, Mat PcaFace, Mat Mean);
int run_cuda_preparation( Mat Image, Mat PcaFace, Mat mean, float *Q);

int run_cuda_Cal_test_pic_Q( Mat Image, Mat PcaFace, Mat Mean, float *q_test);
float Q_compare(float *Q, int width_pca, int num_images);
float Q_compare_test(float *Q, float *q_test, int width_pca, int num_images);
//int run_cuda_deinitialization();
