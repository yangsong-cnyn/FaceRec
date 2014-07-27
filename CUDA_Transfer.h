#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "math.h"

using namespace cv;

void run_cuda(IplImage *frame, PCA *decPCA, Mat pRecon, int threshold);