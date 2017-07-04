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


using namespace std;
using namespace cv;


//Task 1.4
double*** compute3DTemplate(double*** hog, const std::vector<int> &dims, int grid_pos_x, int grid_pos_y) {
	//Test if input is valid
	assert(grid_pos_x > 0);
	assert(grid_pos_y > 0);
	assert(grid_pos_y + TEMPLATE_HEIGHT_CELLS < dims.at(0));
	assert(grid_pos_x + TEMPLATE_WIDTH_CELLS < dims.at(1));

	//cout << "Hog-Size = (" << dims.at(0) << "," << dims.at(1) << "," << dims.at(2) << ")" << endl;

	//init array
	double*** featureRepresentation = 0;

	//alloc space for 3D-array
	featureRepresentation = new double**[TEMPLATE_HEIGHT_CELLS];
	for (int i = 0; i < TEMPLATE_HEIGHT_CELLS; i++) {
		featureRepresentation[i] = new double*[TEMPLATE_WIDTH_CELLS];
		for (int j = 0; j < TEMPLATE_WIDTH_CELLS; j++) {
			featureRepresentation[i][j] = new double[HOG_DEPTH];
		}
	}

	for (int y = 0; y < TEMPLATE_HEIGHT_CELLS; y++) {
		for (int x = 0; x < TEMPLATE_WIDTH_CELLS; x++) {
			for (int b = 0; b < HOG_DEPTH; b++) {
				//copy template-part of hog to output array
				//cout << y << " " << x << " " << b << endl;

				double value = hog[y + grid_pos_y][x + grid_pos_x][b];
				featureRepresentation[y][x][b] = value;

			}
		}
	}
	return featureRepresentation;
}

float* compute1DTemplate(double*** hog, const std::vector<int> &dims, int grid_pos_x, int grid_pos_y, int scale) {
	//Test if input is valid
	assert(grid_pos_x >= 0);
	assert(grid_pos_y >= 0);
	assert(grid_pos_y + TEMPLATE_HEIGHT_CELLS <= dims.at(0));
	assert(grid_pos_x + TEMPLATE_WIDTH_CELLS <= dims.at(1));

	//cout << "Hog-Size = (" << dims.at(0) << "," << dims.at(1) << "," << dims.at(2) << ")" << endl;

	//allocate 1D array
	float* featureRepresentation = (float*)malloc(sizeof(float)*(TEMPLATE_HEIGHT_CELLS*TEMPLATE_WIDTH_CELLS*HOG_DEPTH));
	for (int y = 0; y < TEMPLATE_HEIGHT_CELLS; y++) {
		for (int x = 0; x < TEMPLATE_WIDTH_CELLS; x++) {
			for (int b = 0; b < HOG_DEPTH; b++) {
				//copy values from 3D to 1D array
				float value = hog[y + grid_pos_y][x + grid_pos_x][b];
				featureRepresentation[y*(TEMPLATE_WIDTH_CELLS)*HOG_DEPTH + x*HOG_DEPTH + b] = value;
			}
		}
	}
	return featureRepresentation;
}

vector<float*> get1DTemplateFromPos(string filename, Mat points, int* last) {

	vector<float*> out;

	filename = "INRIAPerson\\" + filename;
	Mat img = imread(filename);

	//Mat img2 = showBoundingBox(img, filename);

	std::vector<int> bboxes = getBoundingBoxes(filename);

	int pos = 0;
	while (bboxes.size() - pos > 3)
	{
		//cout << bboxes[0] << " " << bboxes[1] << " " << bboxes[2] << " " << bboxes[3] << " " << endl;

		//verzieht es momentan noch, evtl verbessern
		int half_height = (bboxes[pos + 3] - bboxes[pos + 1]) / 2;
		int diff = half_height - (bboxes[pos + 2] - bboxes[pos + 0]);
		//cout << "diff:" << diff << endl;

		int x1, y1, height, width;
		if (bboxes[pos + 0] - 8 - diff/2 >= 0) {
			x1 = bboxes[pos + 0] - 8 - diff/2;
		}
		else {
			x1 = 0;
		}
		if (bboxes[pos + 1] - 8 >= 0) {
			y1 = bboxes[pos + 1] - 8;
		}
		else {
			y1 = 0;
		}
		if (bboxes[pos + 2] + 16 + diff < img.cols) {
			width = bboxes[pos + 2] - bboxes[pos + 0] + 16 + diff;
		}
		else {
			width = img.cols - x1;
		}
		if (bboxes[pos + 3] + 16 < img.rows) {
			height = bboxes[pos + 3] - bboxes[pos + 1] + 16;
		}
		else {
			height = img.rows - y1;
		}
		Rect rect(x1, y1, width, height);
		Mat img_croped = img(rect);
		//imshow("croped", img_croped);

		Mat img_scaled(TEMPLATE_HEIGHT + 16, TEMPLATE_WIDTH + 16, CV_8UC3, Scalar(0, 0, 0));
		resize(img_croped, img_scaled, img_scaled.size(), 0, 0, INTER_LINEAR);
		//imshow("scaled", img_scaled);
		//cout << "Scaled: " << img_scaled.size().width << " " << img_scaled.size().height << endl;

		//imshow("bild", img2);

		vector<int> dims;
		double*** HoG = computeHoG(img_scaled, CELL_SIZE, dims);
		//cout << dims[0] << " " << dims[1] << endl;
		Mat grad = visualizeGradOrientations(HoG, dims);
		//imshow("Grad", grad);
		//waitKey();


		float* templateHoG = compute1DTemplate(HoG, dims, 0, 0, 0);

		for (int k = 0; k < points.cols; k++) {			
			points.at<float>(*last, k) = templateHoG[k];
		}
		//cout << "added at:" << *last << " first=" << points.at<float>(*last, 0) << endl;
		(*last)++;
		pos += 4;
		//cout << "added one at " << *last << endl;
	}

	return out;
}


