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

#include "trainSVM.h"
#include "hog.h"
#include "featureExtraction.h"
#include "main.h"
#include "utils.h"


using namespace std;
using namespace cv;


void firstStepTrain() {
	char* filename = "SVM.xml";

	int N = 1000;

	Mat points = createFirstSet(N);
	Mat labels = createFirstLabels(N);

	//Output of points and labels
	
	for (int i = 0; i < 2 * N; i++) {
		//cout << i << " - Label: " << labels.at<float>(i,0) << endl;
		for (int j = 0; j < points.cols; j++) {
			//cout << points.at<float>(i, j);
		}
	}

	// Train with SVM
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	//params.C = 0.01; //best option according to Dalal and Triggs
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100000, 1e-6);

	cout << "Training..." << endl;
	CvSVM SVM;
	SVM.train_auto(points, labels, Mat(), Mat(), params);
	SVM.save(filename);
	cout << "finished training" << endl << endl;
}

Mat createFirstSet(int N) {
	Mat points(2*N, TEMPLATE_WIDTH_CELLS*TEMPLATE_HEIGHT_CELLS, CV_32FC1); //has size = 2*N (height) and Template-width_CELLS * template-height_cells (width)
	//iterate over i = 2*N are the rows -> Each row represents one file=picture

	cout << "Read in Data for SVM ... " << endl;
	//positiv
	string line_pos;
	ifstream myfile_pos("INRIAPerson\\train\\pos.lst");
	for (int i = 0; i < points.rows / 2; i++) {

		getline(myfile_pos, line_pos);
		float* templateHoG;
		templateHoG = getTemplate(line_pos, true);
		
		//copy values of template to Matrix
		for (int j = 0; j < points.cols; j++) {
			points.at<float>(i, j) = templateHoG[j];
		}

		//cout << "point at i=" << i << " from " << line_pos << endl;
		if (i % 50 == 0) {
			cout << "|";
		}
	}
	myfile_pos.close();

	//negativ
	string line_neg;
	ifstream myfile_neg("INRIAPerson\\train\\neg.lst");
	for (int i = points.rows / 2; i < points.rows; i++) {
		getline(myfile_neg, line_neg);

		float* templateHoG;
		templateHoG = getTemplate(line_neg, false);

		//copy values of template to Matrix
		for (int j = 0; j < points.cols; j++) {
			points.at<float>(i, j) = templateHoG[j];
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

Mat createFirstLabels(int N) {

	Mat labels(2 * N, 1, CV_32FC1);

	//positiv
	for (int i = 0; i < labels.rows / 2; i++) {
		labels.at<float>(i, 0) = -1;
	}

	//negativ
	for (int i = labels.rows / 2; i < labels.rows; i++) {
		labels.at<float>(i, 0) = 1;
	}

	/*for (int i = 0; i < labels.rows; i++) {
		float l = i < N ? 1.0 : -1.0;
		labels.at<float>(i, 0) = l;
	}*/

	return labels;
}

float* getTemplate(string filename, bool positiv) {
	vector<int> dims;
	//folder depends on positiv / negativ
	string folder = "INRIAPerson";		
	string in = folder + "/" + filename;

	double*** HoG = extractHOGFeatures(folder, filename, dims);
	float* Template1D = compute1DTemplate(HoG, dims, 0, 0, 0);
	destroy_3Darray(HoG, dims[0], dims[1]);

	return Template1D;
	//return nullptr;
}