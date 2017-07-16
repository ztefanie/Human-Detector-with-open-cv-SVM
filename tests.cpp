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
#include "main.h"
#include "optimizeSVM.h"
#include "featureExtraction.h"
#include "testSVM.h"

using namespace std;
using namespace cv;


/* Task 1.3 
*
* Test extracting a template at a specific position
*
*/
void test3DTemplate()
{
	//compute HoG of whole picture
	vector<int> dims;
	double*** hog = extractHOGFeatures("INRIAPerson\\Train\\pos", "crop_000607.png", dims);

	//extract template
	vector<int> dims2 = vector<int>(3);
	dims2[0] = TEMPLATE_HEIGHT_CELLS;
	dims2[1] = TEMPLATE_WIDTH_CELLS;
	dims2[2] = HOG_DEPTH;
	double*** featureTemplate = compute3DTemplate(hog, dims, 10, 25);
	Mat out2 = visualizeGradOrientations(featureTemplate, dims2);
	imshow("Hog of template at given poistion", out2);

	//clean up
	destroy_3Darray(hog, dims[0], dims[1]);
	destroy_3Darray(featureTemplate, dims2[0], dims2[1]);
}

/* Task 1.3 / 1.4
*
* Shows HoG of a picture
*
*/
void testHog()
{
	vector<int> dims;
	double*** hog = extractHOGFeatures("INRIAPerson\\Train\\pos", "crop_000607.png", dims);
	Mat out = visualizeGradOrientations(hog, dims);
	imshow("Grad", out);
	destroy_3Darray(hog, dims[0], dims[1]);
}

/*
* Test sizes of HoGs of small images
*
*/
void testHogSmallTestImg()
{
	getchar();
	vector<int> dims, dims2, dims3;
	
	double*** hog = extractHOGFeatures("INRIAPerson\\test_64x128_H96\\pos", "crop_000001a.png", dims);
	double*** hog2 = extractHOGFeatures("INRIAPerson\\train_64x128_H96\\pos", "crop_000010a.png", dims2);
	double*** hog3 = extractHOGFeatures("INRIAPerson\\Test\\pos", "crop_000001.png", dims3);
	
	Mat out = visualizeGradOrientations(hog, dims);
	Mat out2 = visualizeGradOrientations(hog2, dims2);
	Mat out3 = visualizeGradOrientations(hog3, dims3);

	imshow("Gradients Test-Image", out);
	imshow("Gradients Train-Image", out2);
	imshow("Gradients Full-size", out3);

	cout << "Dims of HoG (Test): " << dims[0] << ", " << dims[1] << ", " << dims[2] << endl;
	cout << "Dims of HoG (Train): " << dims2[0] << ", " << dims2[1] << ", " << dims2[2] << endl;
	cout << "Dims of HoG (720x491): " << dims3[0] << ", " << dims3[1] << ", " << dims3[2] << endl;
	cout << "Dims of template: " << TEMPLATE_HEIGHT_CELLS << ", " << TEMPLATE_WIDTH_CELLS << ", " << HOG_DEPTH << endl << endl;

	destroy_3Darray(hog, dims[0], dims[1]);
	destroy_3Darray(hog2, dims2[0], dims2[1]);
	destroy_3Darray(hog3, dims3[0], dims3[1]);
}

/* Task 1.1
*
* Test drawing BoundingBoxes
*
*/
void testDrawBoundingBox()
{
	String file = "INRIAPerson\\Train\\pos\\crop_000607.png";
	Mat out = imread(file);
	showBoundingBox(out, file);
	imshow("BoundingBox", out);
	waitKey();
	destroyAllWindows();
}

/* Task 1.2
*
* Testing the overlap computation for some test cases
*
*/
void testOverlapBoundingBox()
{
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

/*
* Changes degree to radient for gradient visualisation
*
*/
double toRadiant(double degree)
{
	assert(degree >= 0 && degree <= 360.);
	double rad = (2 * degree * M_PI) / ((double)360);
	return rad;
}

/*
* Visualize gradients for testing purpose
*
*/
Mat visualizeGradOrientations(double*** hog, vector<int>& dims)
{
	assert(dims.size() == 3);

	int cellRows = dims.at(0);
	int cellCols = dims.at(1);
	int bins = dims.at(2);

	Mat img_out(cellRows * CELL_SIZE, cellCols * CELL_SIZE, CV_8UC1, Scalar(0));

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
				if (value != 0)
				{
					int degree = ((b * 180) / bins) + int(0.5 * (180. / bins));
					double gradDir = toRadiant(degree);

					int centerX = j * CELL_SIZE + CELL_SIZE / 2;
					int centerY = i * CELL_SIZE + CELL_SIZE / 2;

					int length = CELL_SIZE;

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

/* Task 1.5
*
* Visualizes the downscaling process
*
*/
void testDownScale() {
	String file = "INRIAPerson/Train/pos/crop_000607.png";
	Mat img = imread(file);
	int count = 0;

	if (img.empty()) {
		std::cout << "Error: no Image" << endl;
		system("pause");
		return;
	}

	double scale = pow(2.0, 1.0 / LAMBDA);

	double akt_width = img.cols;
	double akt_height = img.rows;
	int int_akt_height = floor(akt_height);
	int int_akt_width = floor(akt_width);
	double hig_scale = 1;

	while (floor(akt_width) >= TEMPLATE_WIDTH && floor(akt_height) >= TEMPLATE_HEIGHT) {
		if (count % LAMBDA == 0) {
			double help = pow(2, count / LAMBDA);
			akt_width = img.cols / help;
			akt_height = img.rows / help;
		}
		else {
			akt_width = akt_width / scale;
			akt_height = akt_height / scale;
		}
		int_akt_height = floor(akt_height);
		int_akt_width = floor(akt_width);
		Mat m(int_akt_height, int_akt_width, CV_8UC3, Scalar(0, 0, 0));
		resize(img, m, Size(int_akt_width, int_akt_height));
		cout << int_akt_height << " " << int_akt_width << endl;

		vector<Point> temp_pos;
		vector<Point> real_temp_pos;
		vector<Point> real_temp_size;
		int counter = 0;
		double calc_size = (int_akt_height * int_akt_width * 2) / (TEMPLATE_WIDTH * TEMPLATE_HEIGHT);
		temp_pos.resize(calc_size);
		real_temp_pos.resize(calc_size);
		real_temp_size.resize(calc_size);
		Mat m2 = img.clone();
		for (int i = 0; i + TEMPLATE_HEIGHT <= int_akt_height; i += floor(TEMPLATE_HEIGHT / 2)) {
			for (int j = 0; j + TEMPLATE_WIDTH <= int_akt_width; j += floor(TEMPLATE_WIDTH / 2)) {
				temp_pos[counter] = Point(j, i);
				real_temp_pos[counter] = Point(j * hig_scale, i * hig_scale);
				real_temp_size[counter] = Point(TEMPLATE_WIDTH * hig_scale + j * hig_scale, TEMPLATE_HEIGHT * hig_scale + i * hig_scale);
				rectangle(m, temp_pos[counter], Point(j + TEMPLATE_WIDTH, i + TEMPLATE_HEIGHT), CV_RGB(255, 255, 0), 1, 8);
				rectangle(m2, real_temp_pos[counter], real_temp_size[counter], CV_RGB(255, 255, 0), 1, 8);
				waitKey();
			}
		}
		rectangle(m, Point(0, 0), Point(TEMPLATE_WIDTH, TEMPLATE_HEIGHT), CV_RGB(0, 0, 255), 1, 8);
		rectangle(m2, Point(0, 0), Point(TEMPLATE_WIDTH * hig_scale, TEMPLATE_HEIGHT * hig_scale), CV_RGB(0, 0, 255), 1, 8);
		imshow("Template postition in the original sized Image", m2);
		imshow("Downscaled Image", m);
		waitKey();
		destroyAllWindows();
		count++;
		hig_scale *= scale;
	}
}