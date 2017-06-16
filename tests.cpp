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
	Mat out = showBoundingBox("crop_000607");
	imshow("BoundingBox", out);
	waitKey();
}

void testOverlapBoundingBox() {

	std::vector<int> truth = std::vector<int>(4, 0);
	truth.at(0) = 0;
	truth.at(1) = 0;
	truth.at(2) = 4;
	truth.at(3) = 4;
	std::vector<int> detected = std::vector<int>(4, 0);
	detected.at(0) = 2;
	detected.at(1) = 2;
	detected.at(2) = 6;
	detected.at(3) = 6;
	cout << "Shoud be 0.1428 is: " << ComputeOverlap(detected, truth) << endl;
	cout << "Shoud be 0.1428 is: " << ComputeOverlap(truth, detected) << endl;

	truth.at(0) = 0;
	truth.at(1) = 0;
	truth.at(2) = 4;
	truth.at(3) = 4;

	detected.at(0) = 2;
	detected.at(1) = 0;
	detected.at(2) = 6;
	detected.at(3) = 4;

	cout << "Shoud be 1/3 is: " << ComputeOverlap(detected, truth) << endl;
	cout << "Shoud be 1/3 is: " << ComputeOverlap(truth, detected) << endl;

	truth.at(0) = 0;
	truth.at(1) = 0;
	truth.at(2) = 4;
	truth.at(3) = 4;

	detected.at(0) = 6;
	detected.at(1) = 6;
	detected.at(2) = 10;
	detected.at(3) = 10;

	cout << "Shoud be 0 is: " << ComputeOverlap(detected, truth) << endl;
	cout << "Shoud be 0 is: " << ComputeOverlap(truth, detected) << endl;

	truth.at(0) = 2;
	truth.at(1) = 0;
	truth.at(2) = 6;
	truth.at(3) = 4;

	detected.at(0) = 0;
	detected.at(1) = 2;
	detected.at(2) = 4;
	detected.at(3) = 6;

	cout << "Shoud be 0.1428 is: " << ComputeOverlap(detected, truth) << endl;
	cout << "Shoud be 0.1428 is: " << ComputeOverlap(truth, detected) << endl;
}