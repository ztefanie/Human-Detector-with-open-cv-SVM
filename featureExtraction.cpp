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
	assert(grid_pos_y + TEMPLATE_HEIGHT_CELLS < dims.at(0)); //18 = (160/8)-2
	assert(grid_pos_x + TEMPLATE_WIDTH_CELLS < dims.at(1)); //10 = 96/8 -2

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
	float* featureRepresentation = new float[TEMPLATE_HEIGHT_CELLS*TEMPLATE_WIDTH_CELLS*HOG_DEPTH];
	for (int y = 0; y < TEMPLATE_HEIGHT_CELLS; y++) {
		for (int x = 0; x < TEMPLATE_WIDTH_CELLS; x++) {
			for (int b = 0; b < HOG_DEPTH; b++) {
				//copy values from 3D to 1D array
				float value = hog[y + grid_pos_y][x + grid_pos_x][b];
				featureRepresentation[y*TEMPLATE_WIDTH_CELLS*HOG_DEPTH + x*HOG_DEPTH + b] = value;
			}
		}
	}
	return featureRepresentation;
}


