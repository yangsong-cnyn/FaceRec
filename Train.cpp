#include "StdAfx.h"
#include <opencv2\contrib\contrib.hpp>  
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
  
#include <iostream>  
#include <fstream>  
#include <sstream>
#include <string>

using namespace std;
using namespace cv;  

static  Mat norm_0_255(cv::InputArray _src)  
{  
    Mat src = _src.getMat();  
    Mat dst;  
  
    switch(src.channels())  
    {  
    case 1:  
        cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);  
        break;  
    case 3:  
        cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);  
        break;  
    default:  
        src.copyTo(dst);  
        break;  
    }  
  
    return dst;  
}  
  
static void read_List(string folderName, vector<Mat> &images, vector<int> &labels)  
{
	string  fileList = folderName+"list";

	ifstream iFile;
	int imgNum;
	iFile.open(fileList);
	if (!iFile)
	   {
		 return;
	   }
	iFile>>imgNum;

	char    temp[50];
	int     labeltemp;
	for (int i=0; i<imgNum; i++)
	    {
	      if (iFile>>temp && iFile>>labeltemp){
		   string filePath = folderName;
		   string fileName(temp);
		   filePath += fileName;
		   images.push_back(cvLoadImage(filePath.c_str(), 0));
           labels.push_back(labeltemp); 
		  }
	    }
} 