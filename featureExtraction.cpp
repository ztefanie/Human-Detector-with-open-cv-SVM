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
	assert(grid_pos_y + TEMPLATE_HEIGHT_CELLS < dims.at(0)); //18
	assert(grid_pos_x + TEMPLATE_WIDTH_CELLS < dims.at(1)); //10

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


//Task 1.5 - mostly the code from fabio (see a1.5.cpp)
vector<templatePos> multiscaleImg(string file) {

	Mat img = showBoundingBox("crop_000607");
	

	int count = 0;
	vector<templatePos> posTemplates;
	CvSVM SVM;
	SVM.load(SVM_2_LOCATION);

	assert(!img.empty());

	double scale = pow(2.0, 1.0 / LAMBDA);

	double akt_width = img.cols;
	double akt_height = img.rows;
	int int_akt_height = floor(akt_height);
	int int_akt_width = floor(akt_width);
	double hig_scale = 1;

	//scale down every loop
	while (floor(akt_width) >= TEMPLATE_WIDTH && floor(akt_height) >= TEMPLATE_HEIGHT) {
		//octave full
		if (count % LAMBDA == 0) {
			double help = pow(2, count / LAMBDA);
			akt_width = img.cols / help;
			akt_height = img.rows / help;
		}
		else {
			akt_width = akt_width / scale;
			akt_height = akt_height / scale;
		}

		//round
		int_akt_height = floor(akt_height);
		int_akt_width = floor(akt_width);

		//resize
		Mat m(int_akt_height, int_akt_width, CV_8UC3, Scalar(0, 0, 0));
		resize(img, m, m.size(), 0, 0, INTER_CUBIC);
		//cout << int_akt_height << " " << int_akt_width << endl;

		//compute HOG for every size
		vector<int> dims;
		double*** hog = computeHoG(m, CELL_SIZE, dims);

		/*Mat out = visualizeGradOrientations(hog, dims);
		String pic = "Gradients at scale: " + to_string(count);
		imshow(pic, out);

		if (dims.at(0) > TEMPLATE_HEIGHT_CELLS && dims.at(1) > TEMPLATE_HEIGHT_CELLS) {
			double*** featureTemplate = compute3DTemplate(hog, dims, 0, 0);
			vector<int> dims_template{ TEMPLATE_HEIGHT_CELLS,TEMPLATE_WIDTH_CELLS, HOG_DEPTH };
			Mat out = visualizeGradOrientations(featureTemplate, dims_template);
			String pic = "first template at scale: " + to_string(count);
			imshow(pic, out);
		}*/

		vector<Point> temp_pos;
		vector<Point> real_temp_pos;
		vector<Point> real_temp_size;
		int counter = 0;
		double calc_size = (int_akt_height * int_akt_width * 2) / (TEMPLATE_WIDTH * TEMPLATE_HEIGHT);
		temp_pos.resize(calc_size);
		real_temp_pos.resize(calc_size);
		real_temp_size.resize(calc_size);

		if (dims.at(0) > TEMPLATE_HEIGHT_CELLS && dims.at(1) > TEMPLATE_HEIGHT_CELLS) {
			//for (int i = 0; i + TEMPLATE_HEIGHT <= int_akt_height; i += floor(TEMPLATE_HEIGHT / 2)) {
				//for (int j = 0; j + TEMPLATE_WIDTH <= int_akt_width; j += floor(TEMPLATE_WIDTH / 2)) {
			int template_count = 1;
			for (int i = 0; i + TEMPLATE_HEIGHT_CELLS < dims.at(0); i += floor(TEMPLATE_HEIGHT_CELLS / 2)) {
				for (int j = 0; j + TEMPLATE_WIDTH_CELLS < dims.at(1); j += floor(TEMPLATE_WIDTH_CELLS / 2)) {
					//if (count == 5) { //Show only for a specific count (just for testing)
						//if (i_cells + TEMPLATE_HEIGHT_CELLS < dims.at(0)) {
					//cout << i << " " << j << endl;
					
					//visualization
					double*** featureTemplate3D = compute3DTemplate(hog, dims, j, i);
					vector<int> dims_template{ TEMPLATE_HEIGHT_CELLS,TEMPLATE_WIDTH_CELLS, HOG_DEPTH };
					Mat out = visualizeGradOrientations(featureTemplate3D, dims_template);
					String pic = "template at " + to_string(i) + to_string(j);
					//imshow(pic, out);

					//3.1 //enumerate...
					templatePos pos;
					pos.x = j*hig_scale;
					pos.y = i*hig_scale;
					pos.scale = hig_scale;


					//3.2 //feature + detection score: distace form the hyperplane
					float* featureTemplate1D = compute1DTemplate(hog, dims, j, i, 0);

					Mat sampleTest(1, TEMPLATE_WIDTH_CELLS*TEMPLATE_HEIGHT_CELLS*HOG_DEPTH, CV_32FC1);
					//copy values of template to Matrix
					for (int j = 0; j < sampleTest.cols; j++) {
						sampleTest.at<float>(0, j) = featureTemplate1D[j];
					}

					float score = SVM.predict(sampleTest, true);

					//cout << "Number: " << template_count << " xPos: " << j * hig_scale << " yPos: " << i * hig_scale << "Scale = " << hig_scale << " with score: " << score << endl;

					//3.3
					//copy templates with detections of peoples to output
					if (hig_scale >= 3 ) {
						cout << "FOUND" << endl;
						posTemplates.push_back(pos);
						//for testing
						/*Scalar color = Scalar(100, 200, 100);
						Point p1 = Point(j*hig_scale*CELL_SIZE, i*hig_scale*CELL_SIZE);
						Point p2 = Point(j*hig_scale*CELL_SIZE, i*hig_scale*CELL_SIZE+ TEMPLATE_WIDTH);
						Point p3 = Point(j*hig_scale*CELL_SIZE+TEMPLATE_HEIGHT, i*hig_scale*CELL_SIZE);
						Point p4 = Point(j*hig_scale*CELL_SIZE+TEMPLATE_HEIGHT, i*hig_scale*CELL_SIZE + TEMPLATE_WIDTH);
						line(img, p1, p2, color, 1);
						line(img, p1, p3, color, 1);
						line(img, p2, p4, color, 1);
						line(img, p3, p4, color, 1);*/
						real_temp_pos[counter] = Point(j * hig_scale, i * hig_scale);
						real_temp_size[counter] = Point(TEMPLATE_WIDTH * hig_scale + j * hig_scale, TEMPLATE_HEIGHT * hig_scale + i * hig_scale);
						//just for viso:
						//rectangle(img, real_temp_pos[counter], real_temp_size[counter], CV_RGB(255, 255, 0), 1, 8);
					}
					
					//waitKey();
					//destroyAllWindows();
					template_count++;
					//}
				//}
				}
			}
		}
		rectangle(img, Point(0, 0), Point(TEMPLATE_WIDTH * hig_scale, TEMPLATE_HEIGHT * hig_scale), CV_RGB(0, 0, 255), 1, 8);
		count++;
		hig_scale *= scale;
		cout << hig_scale << endl;

		//destroy at end of each scale
		destroy_3Darray(hog, dims[0], dims[1]);
	}
	imshow("file", img);
	return posTemplates;
}

