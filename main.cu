#pragma  once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cv.h>
#include <highgui.h>
#include <stdio.h>

using namespace std;
using namespace cv;

__global__ void Edge(uchar * inMAP, uchar * outMAP) {

    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int Dim = gridDim.x *blockDim.x;
    int offset = x + y *Dim;
    
	if (inMAP[offset] < 127)
	{
	    outMAP[offset] =0;
	}
	else
	{
	    outMAP[offset] =255;
	}
}

extern "C" int run(IplImage *frame, PCA *decPCA, Mat pRecon, int threshold)
{
	uchar     *inMAP;
	uchar     *outMAP;
	uchar     *data = (uchar*)frame->imageData;
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

	return 0;
}