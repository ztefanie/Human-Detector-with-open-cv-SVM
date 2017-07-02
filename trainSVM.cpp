#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <iostream>
#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h> 
#include <stdlib.h>

#include "trainSVM.h"
#include "hog.h"
#include "featureExtraction.h"
#include "main.h"
#include "utils.h"


using namespace std;
using namespace cv;


void firstStepTrain() {

	//Get Sizes of Datasets
	int N_pos = 0;
	std::string line;
	std::ifstream myfile(LIST_POS);
	while (std::getline(myfile, line))
		++N_pos;
	N_pos *= 5;

	int N_neg = 0;
	std::ifstream myfile2(LIST_NEG);
	while (std::getline(myfile2, line))
		++N_neg;
	N_neg *= 10;

	Mat points = createFirstSet(N_pos, N_neg);
	Mat labels = createFirstLabels(N_pos, N_neg);

	// Train with SVM
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.C = 0.01; //best option according to Dalal and Triggs
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100000, 1e-6);

	cout << "Training SVM with " << points.rows << " Datapoints... " << endl;
	CvSVM SVM;
	SVM.train_auto(points, labels, Mat(), Mat(), params);
	SVM.save(SVM_LOCATION);
	cout << "finished training" << endl << endl;
}

Mat createFirstSet(int N_pos, int N_neg) {
	int template_size = (TEMPLATE_WIDTH_CELLS-2)*(TEMPLATE_HEIGHT_CELLS-2)*HOG_DEPTH;
	Mat points(N_pos+N_neg, template_size, CV_32FC1); //has size = 2*N (height) and Template-width_CELLS * template-height_cells (width)
	//iterate over i = 2*N are the rows -> Each row represents one file=picture

	cout << "Read in Data for SVM ... " << endl;
	//positiv
	string line_pos;
	ifstream myfile_pos(LIST_POS);
	for (int i = 0; i < N_pos; i++) {

		float* templateHoG;
		if (i % 5 == 0) {
			getline(myfile_pos, line_pos);
			line_pos.insert(5, "_64x128_H96");
			templateHoG = getTemplate(line_pos, true, true);
		}
		
		//copy values of template to Matrix
		for (int j = 0; j < template_size; j++) {
			points.at<float>(i, j) = templateHoG[j];
		}

		//cout << "point at i=" << i << " from " << line_pos << endl;
		if (i % 250 == 0) {
			cout << "|";
		}
	}
	myfile_pos.close();

	//negativ
	string line_neg;
	ifstream myfile_neg(LIST_NEG);
	for (int i = N_pos; i < points.rows/10; i++) {
		getline(myfile_neg, line_neg);
		line_neg.insert(5, "_64x128_H96");
		//for each negativ 10 templates
		for (int k = 0; k < 10; k++) {
			float* templateHoG;
			templateHoG = getTemplate(line_neg, false, true);

			//copy values of template to Matrix
			for (int j = 0; j < points.cols; j++) {
				points.at<float>(i*10 + k, j) = templateHoG[j];
			}
		}
		//cout << "point at i=" << i << " from " << line_neg << endl;
		if (i % 50 == 0) {
			cout << "|";
		}		
	}
	cout << endl << "finished read in Data for SVM" << endl << endl;
	myfile_neg.close();

	return points;
}

Mat createFirstLabels(int N_pos, int N_neg) {

	Mat labels(N_pos + N_neg, 1, CV_32FC1);

	//positiv
	for (int i = 0; i < N_pos; i++) {
		labels.at<float>(i, 0) = -1;
	}

	//negativ
	for (int i = N_pos / 2; i < labels.rows; i++) {
		labels.at<float>(i, 0) = 1;
	}

	return labels;
}

float* getTemplate(string filename, bool positiv, bool training) {
	vector<int> dims;
	string folder = "INRIAPerson";		

	String get = folder + "\\" + filename;

	Mat img = imread(get);
	Mat templateMat;

	if (positiv && training) {
		Rect rect(16, 16, 64, 128);		
		templateMat = img(rect);
	}
	else if (positiv && !training) {
		Rect rect(3, 3, 64, 128);
		templateMat = img(rect);
	}
	else {
		srand(time(NULL));
		int height_new = rand() % (img.size().height - 128) + 128;
		double new_size = height_new / (double)img.size().height;
		resize(img, img, Size(), new_size, new_size, 1);
		int offsetY = rand() % (img.size().width / 8 - 2 - TEMPLATE_HEIGHT_CELLS);
		int offsetX = rand() % (img.size().height / 8 - 2 - TEMPLATE_WIDTH_CELLS);
		Rect rect(offsetX, offsetY, 64, 128);
		templateMat = img(rect);
	}
		
	double*** HoG = computeHoG(templateMat, CELL_SIZE, dims);
	float* Template1D = compute1DTemplate(HoG, dims, 0, 0, 0);

	destroy_3Darray(HoG, dims[0], dims[1]);

	return Template1D;
}