#define _USE_MATH_DEFINES

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>

#include "tests.h"
#include "utils.h"
#include "hog.h"

using namespace std;
using namespace cv;

void testHog() {
	vector<int> dims;
	double*** hog = extractHOGFeatures("INRIAPerson\\Train\\pos", "crop_000607.png", dims);
	Mat out = visualizeGradOrientations(hog, dims);
	imshow("Grad", out);
}

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
double toRadiant(double degree)
{
	assert(degree >= 0 && degree <= 360.);

	double rad = (2 * degree * M_PI) / (double)360;

	return rad;
}

Mat visualizeGradOrientations(double*** hog, vector<int> &dims) {

	assert(dims.size() == 3);

	int cellRows = dims.at(0);
	int cellCols = dims.at(1);
	int bins = dims.at(2);

	Mat img_out(cellRows * cell_size, cellCols * cell_size, CV_8UC1, Scalar(0));

	for (int i = 0; i < cellRows; i++)
	{
		for (int j = 0; j < cellCols; j++)
		{
			double max = -1, min = -1;
			for (int b = 0; b < bins; b++)
			{
				double value = hog[i][j][b];
				if (value == 0)
					continue;
				if (max == -1 || value > max)
					max = value;
				if (min == -1 || value < min)
					min = value;
			}
			//check if gradient is empty
			if (min == -1 || max == -1)
			{
				continue;
			}

			for (int b = 0; b < bins; b++)
			{
				double value = hog[i][j][b];
				if (value != 0) {

					int degree = ((b * 180) / bins) + int(0.5 * (180. / bins));
					double gradDir = toRadiant(degree);

					int centerX = j * cell_size + cell_size / 2;
					int centerY = i * cell_size + cell_size / 2;

					int length = cell_size;

					int xOffset = int((cos(gradDir) * length) / 2);
					int yOffset = int((sin(gradDir) * length) / 2);

					Point P1(centerX + xOffset, centerY + yOffset);
					Point P2(centerX - xOffset, centerY - yOffset);

					uchar strength = uchar(255. * (value / max));

					line(img_out, P1, P2, Scalar(strength));
				}
			}
		}
	}
	return img_out;
}
