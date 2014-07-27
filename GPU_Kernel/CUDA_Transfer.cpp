#include "stdafx.h"
#include "CUDA_Transfer.h"
#include <math.h>
using namespace cv;

extern "C" void run( IplImage *frame, Mat eigenVector, Mat eigenValue, Mat mean);
extern "C" int run_prepartion( Mat Image, Mat PcaFace, Mat Mean, float *Q);
extern "C" float getnorm( float *Q, int rows, int colomns);
extern "C" int run_initialization( Mat Image, Mat PcaFace, Mat Mean);
extern "C" int Cal_test_pic_Q( Mat Image, Mat PcaFace, Mat Mean, float *q_test);
extern "C" int run_deinitialization(void);

int run_cuda_initialization( Mat Image, Mat PcaFace, Mat Mean)
{
	run_initialization(Image, PcaFace, Mean);
	return 0;
}

int run_cuda_preparation( Mat Image, Mat PcaFace, Mat mean, float *Q)
{
	run_prepartion( Image, PcaFace, mean, Q);
	return 0;
}

/*int run_cuda_deinitialization()
{
	run_deinitialization();
	return 0;
}*/

int run_cuda_Cal_test_pic_Q( Mat Image, Mat PcaFace, Mat Mean, float *q_test)
{
	Cal_test_pic_Q( Image, PcaFace, Mean, q_test);
	return 0;
}

float Q_compare(float *Q, int width_pca, int num_images)
{
	float delta_Qmax = 0.0;
	float sum = 0.0;
	for (int i = 0; i < num_images - 1; i++)
	{
		for (int j = i+1; j < num_images; j++)
		{
			for (int k =0; k < width_pca; k++)
			{
				sum += pow(Q[k + i * width_pca] - Q[k + j * width_pca], 2);
			}
			sum = sqrtf(sum);
			if (sum > delta_Qmax)
				delta_Qmax = sum;
		}
	}
	return delta_Qmax;
}

float Q_compare_test(float *Q, float *q_test, int width_pca, int num_images)
{
	float delta_Qmin = 999999999.0;
	float sum =0.0;
	for (int i = 0; i < num_images; i++)
	{
		for (int k = 0; k < width_pca; k++)
		{
			sum += pow(Q[k + i * width_pca] - q_test[k], 2);
		}
		sum = sqrtf(sum);
		if (sum < delta_Qmin)
			delta_Qmin = sum;
	}
	return delta_Qmin;
}
