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
#include "optimizeSVM.h"
#include "hog.h"
#include "featureExtraction.h"
#include "main.h"
#include "utils.h"


using namespace std;
using namespace cv;

int iterations = 100000;


/* Task 2
*
* trains or retrains SVM
*
* @param retraining: bool if the SVM should be trained or retrained
*
*/
void SVMtrain(bool retraining) {

	int factor_pos = 5;
	int factor_neg = 10;
	if (retraining) {
		factor_pos = 5;
		factor_neg = 8;
	}

	std::string line;
	int N_pos = 2474;
	N_pos *= factor_pos;

	int N_neg = 0;
	std::ifstream myfile2(LIST_NEG);
	while (std::getline(myfile2, line))
		++N_neg;
	N_neg *= factor_neg;

	Mat points = createFirstSet(N_pos, N_neg, factor_pos, factor_neg);
	Mat labels = createFirstLabels(N_pos, N_neg);

	Mat all_neg;
	if (retraining) {
		//Mat hardNegatives = find_hardNegatives();
		//hardNegatives.copyTo(points(Rect(0, points.rows - (MAX_HARD_NEG+1), hardNegatives.cols, hardNegatives.rows)));
		Mat hardNegatives = find_hardNegatives();
		vconcat(points, hardNegatives, all_neg);

		Mat label_neg(1, 1, CV_32FC1);
		label_neg.at<float>(0, 0) = 1.0;
		for (int i = 0; i < hardNegatives.rows; i++) {
			labels.push_back(label_neg);
		}
	}


	// Train with SVM
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.C = 0.01; //best option according to Dalal and Triggs
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, iterations, 1e-6);


	CvSVM SVM;
	if (retraining) {
		cout << "Training SVM with " << all_neg.rows << " Datapoints... (" << N_pos << ", " << all_neg.rows - N_pos << ") since: " << getTimeFormatted()<< endl;
		SVM.train_auto(all_neg, labels, Mat(), Mat(), params);
		SVM.save(SVM_2_LOCATION);
	}
	else {
		cout << "Training SVM with " << points.rows << " Datapoints... (" << N_pos << ", " << N_neg << ") since: " << getTimeFormatted() << endl;
		SVM.train_auto(points, labels, Mat(), Mat(), params);
		SVM.save(SVM_LOCATION);
	}
	cout << "finished training at " << getTimeFormatted() << endl << endl;
}


/* Task 2
*
* creates input data for the SVM
* 
* @returns: Mat with all input data
* @params N_pos: Number of positiv input data
* @params N_neg: Number of negativ input pictures
* @params factor_pos: factor for oversampling the positiv data
* @params factor_neg: factor how much negativ templates should be extracted for each negativ image
*
*/
Mat createFirstSet(int N_pos, int N_neg, int factor_pos, int factor_neg) {
	int template_size = (TEMPLATE_WIDTH_CELLS)*(TEMPLATE_HEIGHT_CELLS)*HOG_DEPTH;
	Mat points(N_pos + N_neg, template_size, CV_32FC1); //has size = 2*N (height) and TEMPLATE_WIDTH_CELLS*TEMPLATE_HEIGHT_CELLS*HOG_DEPTH
	Mat points_all_pos(N_pos, template_size, CV_32FC1);
	Mat points_temp_pos(N_pos / factor_pos, template_size, CV_32FC1);
	Mat points_temp_neg(N_neg, template_size, CV_32FC1);

	cout << "Read in Data for SVM ... " << endl;
	
	//positiv
	cout << "positives ";
	string line_pos;
	ifstream myfile_pos(LIST_POS);
	int last = 0;
	int i = 0;
	while (getline(myfile_pos, line_pos)) {
		float* templateHoG;
		vector<float*> templates = get1DTemplateFromPos(line_pos, points_temp_pos, &last, false);
		if (i % 10 == 0) {
			cout << "|";
		}
		i++;
	}
	myfile_pos.close();


	//negativ
	string line_neg;
	ifstream myfile_neg(LIST_NEG);
	cout << endl << "negatives: ";
	for (int i = N_pos; i < points.rows; i += factor_neg) {
		getline(myfile_neg, line_neg);
		//for each negativ picture use #factor_neg templates
		for (int k = 0; k < factor_neg; k++) {
			float* templateHoG;
			templateHoG = getTemplate(line_neg);
			//copy values of template to Matrix
			for (int j = 0; j < points.cols; j++) {
				points_temp_neg.at<float>(i + k - N_pos, j) = templateHoG[j];
			}
		}
		if ((i-N_pos) % (20* factor_neg) == 0) {
			cout << "|";
		}
	}
	cout << endl << "finished read in Data for SVM" << endl << endl;
	myfile_neg.close();

	//Duplicate positiv input data for balances size
	for (int p = 0; p < factor_pos; p++) {
		Mat dst_roi = points_all_pos(Rect(0, p*(N_pos / factor_pos), points_temp_pos.cols, points_temp_pos.rows));
		points_temp_pos.copyTo(dst_roi);
	}

	//Concat positiv and negativ to one Mat
	vconcat(points_all_pos, points_temp_neg, points);

	return points;
}

/* Task 2
*
* Creates Labels for the training
*
* @returns: Mat with all the labels
* @params N_pos: Number of positiv input data
* 
*/
Mat createFirstLabels(int N_pos, int N_neg) {

	Mat labels(N_pos + N_neg, 1, CV_32FC1);

	//positiv
	for (int i = 0; i < N_pos; i++) {
		labels.at<float>(i, 0) = -1.0;
	}

	//negativ
	for (int i = N_pos; i < labels.rows; i++) {
		labels.at<float>(i, 0) = 1.0;
	}

	return labels;
}

/*
* Gets random template from an images, used for extracting negativ trainingsdata
*
* @returns: template es a float array
* @param filename: filepath to the picture, where the template should be extracted from
*
*/
float* getTemplate(string filename) {
	vector<int> dims;
	string folder = "INRIAPerson";

	String get = folder + "\\" + filename;
	Mat img = imread(get);

	double*** HoG;
	float* Template1D;

	int height_new = rand() % (img.size().height - (128 + 16)) + (128 + 16);
	double new_size = height_new / (double)img.size().height;
	resize(img, img, Size(), new_size, new_size, 1);

	HoG = computeHoG(img, CELL_SIZE, dims);
	Mat hog_pis = visualizeGradOrientations(HoG, dims);

	int offsetX = 0;
	int offsetY = 0;
	if (dims[0] > TEMPLATE_HEIGHT_CELLS && dims[1] > TEMPLATE_WIDTH_CELLS) {
		offsetX = rand() % (dims[0] - TEMPLATE_HEIGHT_CELLS) + 1;
		offsetY = rand() % (dims[1] - TEMPLATE_WIDTH_CELLS) + 1;
	}
	Template1D = compute1DTemplate(HoG, dims, offsetY, offsetX, 0);
	destroy_3Darray(HoG, dims[0], dims[1]);

	return Template1D;
}