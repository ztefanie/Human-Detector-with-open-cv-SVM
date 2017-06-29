#define _USE_MATH_DEFINES

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>

#include "DET.h"
#include "tests.h"
#include "utils.h"
#include "hog.h"
#include "main.h"
#include "optimizeSVM.h"


using namespace std;
using namespace cv;

