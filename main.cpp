#define _USE_MATH_DEFINES

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <time.h>

#include "hog.h"

using namespace std;
using namespace cv;


int main(int argc, char* argv[]) {
	Mat img;
	vector<int> dims;
	computeHoG(img, 3, dims);
	return 0;
}