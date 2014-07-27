#pragma  once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
 #include "device_functions.h"
#include "cuda.h"
#include <iostream>
#include <Windows.h>
#include <time.h>

using namespace std;
using namespace cv;

	uchar *image;
	uchar *pcaface;
	uchar *mean;
	float *q;
	int test_km;

	uchar *gpu_image;
	uchar *gpu_pcaface;
	uchar *gpu_mean;
	float *gpu_distance;
	float *gpu_Q;
	dim3   grid1(256, 1);
	dim3   block1(256, 1);

__global__ void Edge(uchar * inMAP, uchar * outMAP) {

    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int Dim = gridDim.x *blockDim.x;
    int offset = x + y *Dim;

}

__global__ void MatrixMul(uchar *M, float *N, float *P, int num_pca) 
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
	int offset = x;
	__shared__ uchar M_temp[256];
	__shared__ uchar N_temp[256];

	M_temp[threadIdx.x] = 0;
	N_temp[threadIdx.x] = 0;
	__syncthreads();
	for (int i = 0; i < num_pca; i++)
		M_temp[threadIdx.x] = M[i, offset];
	N_temp[threadIdx.x] = N[offset];
	__syncthreads();
	for(int i = 0; i < num_pca; i++)
		atomicAdd(&P[i], (float) M_temp[threadIdx.x] * N_temp[threadIdx.x]);
}

__global__ void MatrixTranverse(float *M)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int Dim = gridDim.x *blockDim.x;
    int offset = x + y *Dim; //global ID
	M[offset] = M[x * Dim + y];
}

__global__ void DCal(uchar *TestImage, uchar *AverageFace, float *distance) 
{
 //   int offset = threadIdx.x + blockDim.x*blockIdx.x;
/*	int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int Dim = gridDim.x *blockDim.x;
    int offset = x + y *Dim;
	distance[ offset ] =  (float) (TestImage[ offset ] - AverageFace[ offset ]); */
}

extern "C" int run_initialization( Mat Image, Mat PcaFace, Mat Mean)
{
	uchar *image = (uchar *) Image.data;
	uchar *pcaface = (uchar *) PcaFace.data;
	uchar *mean = (uchar *) Mean.data;
	int PcaLength = PcaFace.rows;
	int PcaWidth = PcaFace.cols;
	size_t pitch;
	size_t host_orig_pitch = PcaWidth * sizeof(uchar);
	/**************************** start GPU initilzations **************************/
	cudaError_t cudaStatus;    
	cudaStatus = cudaSetDevice(0);
	assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaMalloc((void**)&gpu_mean, Image.rows * Image.cols * sizeof(uchar));
    assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaMemcpy(gpu_mean, mean, Mean.rows * Mean.cols * sizeof(uchar), cudaMemcpyHostToDevice);
	assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaMalloc((void**)&gpu_Q, 1 * PcaFace.rows * sizeof(float));
	assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaMalloc((void**)&gpu_distance, Image.rows * Image.cols * sizeof(char));
	assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaMallocPitch(&gpu_pcaface, &pitch, PcaWidth * sizeof(uchar), PcaLength * sizeof(uchar));
    assert(cudaStatus == cudaSuccess);

	cudaMemcpy2D(gpu_pcaface, pitch, pcaface, host_orig_pitch, PcaWidth, PcaLength, cudaMemcpyHostToDevice);
    assert(cudaStatus == cudaSuccess);

	/*************************** CPU initializations ******************************/
	q = (float*) malloc(sizeof(float) * 1 * PcaFace.rows);
	return 0;
}

extern "C" int run_prepartion( Mat Image, Mat PcaFace, Mat Mean, float *Q)
{
	uchar *image = (uchar *) Image.data;
	uchar *pcaface = (uchar *) PcaFace.data;
	uchar *mean = (uchar *) Mean.data;
	uchar *image1;

	float *test_distance;
	cudaError_t cudaStatus;
//	int test_km = 0;

	test_distance = (float *)malloc(sizeof(float)* Image.rows * Image.cols);
	cudaStatus = cudaMalloc((void**)&gpu_image, Image.rows * Image.cols * sizeof(uchar));
	image1 = (uchar *)malloc(sizeof(uchar) * Image.rows * Image.cols);
	assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaMemcpy(gpu_image, image, Image.rows * Image.cols * sizeof(uchar), cudaMemcpyHostToDevice);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMemcpy(image1, gpu_image, Image.rows * Image.cols * sizeof(uchar), cudaMemcpyDeviceToHost);
	assert(cudaStatus == cudaSuccess);

    /*********************** GPU Calculation: Calculating D *************************/
	DCal<<<grid1,block1>>>(gpu_image, gpu_mean, gpu_distance);
	cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaDeviceSynchronize();
    assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaMemcpy(test_distance, gpu_distance, Image.rows * Image.cols * sizeof(uchar), cudaMemcpyDeviceToHost);
	assert(cudaStatus == cudaSuccess);
	/*********************** GPU Calculation: Calculating Q**************************/
	MatrixMul<<<grid1,block1>>>(gpu_pcaface, gpu_distance, gpu_Q, PcaFace.rows);
	cudaStatus = cudaMemcpy(q, gpu_Q, 1 * PcaFace.rows * sizeof(float), cudaMemcpyDeviceToHost);
    assert(cudaStatus == cudaSuccess);

	for (int j=0; j<PcaFace.rows; j++)
		{
			Q[test_km + j] = q[j];
		}
		test_km += PcaFace.rows;
		return 0;
}

extern "C" int Cal_test_pic_Q( Mat Image, Mat PcaFace, Mat Mean, float *q_test)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(gpu_image, image, Image.rows * Image.cols * sizeof(uchar), cudaMemcpyHostToDevice);
	assert(cudaStatus == cudaSuccess);
    /*********************** GPU Calculation: Calculating D *************************/
	DCal<<<grid1,block1>>>(gpu_image, gpu_mean, gpu_distance);
	cudaStatus = cudaDeviceSynchronize();
    assert(cudaStatus == cudaSuccess);

	/*********************** GPU Calculation: Calculating Q**************************/
	MatrixMul<<<grid1,block1>>>(gpu_pcaface, gpu_distance, gpu_Q, PcaFace.cols);
	cudaStatus = cudaMemcpy(q_test, gpu_Q, Image.rows * PcaFace.rows * sizeof(float), cudaMemcpyDeviceToHost);
    assert(cudaStatus == cudaSuccess);
	return 0;
}

extern "C" int deinitialization(void)
{
	cudaFree(gpu_image);
	cudaFree(gpu_pcaface);
	cudaFree(gpu_mean);
	cudaFree(gpu_distance);
	cudaFree(gpu_Q);
	free(q);
	return 0;
}

extern "C" void run(IplImage *frame, Mat eigenVector, Mat eigenValue, Mat mean)
{
	uchar     *inMAP;
	uchar     *outMAP;
	uchar	  *distance;
	uchar	  *data = (uchar*)frame->imageData;
	uchar	  *eigen_vector = (uchar *) eigenVector.data;
	int		  H_eigenVector = eigenVector.rows;
	int		  W_eigenVector = eigenVector.cols;
	///////////////////////////////////////////
	cudaError_t cudaStatus;    
	cudaStatus = cudaSetDevice(0);
    assert(cudaStatus == cudaSuccess);
	
	cudaStatus = cudaMalloc((void**)&inMAP, frame->height*frame->width* sizeof(uchar));
    assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaMalloc((void**)&outMAP, frame->height*frame->width* sizeof(uchar));
    assert(cudaStatus == cudaSuccess);
	/////////////////////////////////////////////
	int size = frame->height* frame->width* sizeof(uchar);
    cudaStatus = cudaMemcpy(inMAP, data, size, cudaMemcpyHostToDevice);
    assert(cudaStatus == cudaSuccess);

	dim3 grid(frame->width,frame->height);
    dim3 block(1,1);

	Edge<<<grid, block>>>(inMAP, outMAP);
	cudaStatus = cudaDeviceSynchronize();
    assert(cudaStatus == cudaSuccess);

	cudaStatus = cudaMemcpy(data, outMAP, size, cudaMemcpyDeviceToHost);
    assert(cudaStatus == cudaSuccess);
}