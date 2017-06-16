#define _USE_MATH_DEFINES

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>

#include "hog.h"
#include "utils.h"
#include "featureExtraction.h"
#include "trainSVM.h"
#include "tests.h"

using namespace std;
using namespace cv;


int main(int argc, char* argv[]) {

	//Task 1.1
	testDrawBoundingBox();

	return 0;
}