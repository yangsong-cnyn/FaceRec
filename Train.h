#pragma once
#include <opencv2\contrib\contrib.hpp>  
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
  
#include <iostream>  
#include <fstream>  
#include <sstream>
#include <string>

using namespace std;
using namespace cv; 

void read_List(string filename, vector<Mat> &images, vector<int> &labels);