#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>

#include "tests.h"
#include "utils.h"

using namespace std;
using namespace cv;

void testDrawBoundingBox() {
	Mat out = showBoundingBox("crop_000010");
	imshow("BoundingBox", out);
	waitKey();
}