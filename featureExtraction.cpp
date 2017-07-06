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


/*
* Computes a HoG-array of a template at a given position
*
* @returns HoG-array of template
* @param hog: HoG of the whole picture
* @param dims: dimension of the HoG of the whole picture
* @param grid_pos_x and grid_pos_y: position which template should be extracted
*
*/
double*** compute3DTemplate(double*** hog, const std::vector<int> &dims, int grid_pos_x, int grid_pos_y) {
	//Test if input is valid
	assert(grid_pos_x > 0);
	assert(grid_pos_y > 0);
	assert(grid_pos_y + TEMPLATE_HEIGHT_CELLS < dims.at(0));
	assert(grid_pos_x + TEMPLATE_WIDTH_CELLS < dims.at(1));

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

	//fill with values from the HoG
	for (int y = 0; y < TEMPLATE_HEIGHT_CELLS; y++) {
		for (int x = 0; x < TEMPLATE_WIDTH_CELLS; x++) {
			for (int b = 0; b < HOG_DEPTH; b++) {
				double value = hog[y + grid_pos_y][x + grid_pos_x][b];
				featureRepresentation[y][x][b] = value;

			}
		}
	}

	return featureRepresentation;
}

/*
* Computes a 1D-array of a template at a given position
*
* @returns 1D-array of the HoGs of a template
* @param hog: HoG of the whole picture
* @param dims: dimension of the HoG of the whole picture
* @param grid_pos_x and grid_pos_y: position which template should be extracted
*
*/
float* compute1DTemplate(double*** hog, const std::vector<int> &dims, int grid_pos_x, int grid_pos_y, int scale) {
	//Test if input is valid
	assert(grid_pos_x >= 0);
	assert(grid_pos_y >= 0);
	assert(grid_pos_y + TEMPLATE_HEIGHT_CELLS <= dims.at(0));
	assert(grid_pos_x + TEMPLATE_WIDTH_CELLS <= dims.at(1));

	//allocate 1D array
	float* featureRepresentation = (float*)malloc(sizeof(float)*(TEMPLATE_HEIGHT_CELLS*TEMPLATE_WIDTH_CELLS*HOG_DEPTH));
	
	//copy values from 3D to 1D array
	for (int y = 0; y < TEMPLATE_HEIGHT_CELLS; y++) {
		for (int x = 0; x < TEMPLATE_WIDTH_CELLS; x++) {
			for (int b = 0; b < HOG_DEPTH; b++) {
				float value = hog[y + grid_pos_y][x + grid_pos_x][b];
				featureRepresentation[y*(TEMPLATE_WIDTH_CELLS)*HOG_DEPTH + x*HOG_DEPTH + b] = value;
			}
		}
	}

	return featureRepresentation;
}

/*
* Get templates from positives images, which are marked as persons in the annotations file
*
* @returns vector of 1D template representations
* @param filename: filename of the picture and annotations file
* @points: Mat which takes all the templates for training the SVM
* @param last: the last row in points where a template was added to
* @param show: defines if the pictures (cropped, scaled, skipped) are showed or not
*
*/
vector<float*> get1DTemplateFromPos(string filename, Mat points, int* last, bool show) {

	vector<float*> out;

	filename = "INRIAPerson\\" + filename;
	Mat img = imread(filename);

	std::vector<int> bboxes = getBoundingBoxes(filename);

	int pos = 0;
	//Iterate over all boundingboxes of the picture
	while (bboxes.size() - pos > 3)
	{
		//compute the size an position of the template 
		//depends on the width-to-height ratio of the bounding box and if the bounding box is on the pictures edges
		int half_height = (bboxes[pos + 3] - bboxes[pos + 1]) / 2;
		int diff = half_height - (bboxes[pos + 2] - bboxes[pos + 0]);

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

		//cut out the interesting part of the image
		Rect rect(x1, y1, width, height);
		Mat img_croped = img(rect);
		//scale it down
		Mat img_scaled(TEMPLATE_HEIGHT + 16, TEMPLATE_WIDTH + 16, CV_8UC3, Scalar(0, 0, 0));
		resize(img_croped, img_scaled, img_scaled.size(), 0, 0, INTER_LINEAR);
		//mirror it for better results
		Mat img_mirrored = img_scaled.clone();
		flip(img_scaled, img_mirrored, 1);

		if (show) {
			imshow("original", img);
			imshow("croped", img_croped);
			imshow("scaled", img_scaled);
			imshow("flipped", img_mirrored);					
			waitKey();
		}
		else {
			//compute HoG and 1D-representation for template
			vector<int> dims;
			double*** HoG = computeHoG(img_scaled, CELL_SIZE, dims);
			float* templateHoG = compute1DTemplate(HoG, dims, 0, 0, 0);
			//add it to points
			for (int k = 0; k < points.cols; k++) {
				points.at<float>(*last, k) = templateHoG[k];
			}
			(*last)++;

			//compute HoG and 1D-representation for skipped template
			HoG = computeHoG(img_scaled, CELL_SIZE, dims);
			templateHoG = compute1DTemplate(HoG, dims, 0, 0, 0);
			//add it to points
			for (int k = 0; k < points.cols; k++) {
				points.at<float>(*last, k) = templateHoG[k];
			}
			(*last)++;
			
			//free memory
			free(templateHoG);
			destroy_3Darray(HoG, dims[0], dims[1]);
		}
		pos += 4;
	}

	return out;
}


