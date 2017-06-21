#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <iostream>

#include "trainSVM.h"
#include "hog.cpp"
#include "featureExtraction.h"
#include "main.h"
#include "utils.h"



using namespace std;
using namespace cv;


void firstStepTrain() {
	char* filename = "SVM.xml";

	int N = 20;

	Mat points = createFirstSet(N);
	Mat labels = createFirstLabels(N);

	// Train with SVM
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.C = 0.01; //best option according to Dalal and Triggs
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 6000, 1e-6);

	CvSVM SVM;
	SVM.train_auto(points, labels, Mat(), Mat(), params);
	SVM.save(filename);
}

Mat createFirstSet(int N) {
	Mat points(2*N, TEMPLATE_WIDTH_CELLS*TEMPLATE_HEIGHT_CELLS, CV_64FC1); //has size = 2*N (height) and Template-width_CELLS * template-height_cells (width)
	//iterate over i = 2*N are the rows -> Each row represents one file=picture
	for (int i = 0; i < points.rows; i++) { 
		double* templateHoG;
		if (i < N) { //positiv
			templateHoG = getTemplate(i, true);
		}
		else {	//negativ
			templateHoG = getTemplate(N-i, false);
		}
		
		//copy values of template to Matrix
		for (int j = 0; j < points.cols; j++) {
			points.at<double>(i, j) = templateHoG[j];
		}
	}
	return points;
}

Mat createFirstLabels(int N) {
	Mat labels(2 * N, 1, CV_64FC1);
	for (int i = 0; i < labels.rows; i++) {
		double l = i < N ? 1.0 : -1.0;
		labels.at<double>(i, 0) = l;
	}
	return labels;
}

double* getTemplate(int i, bool positiv) {
	vector<int> dims;
	//folder depends on positiv / negativ
	string folder;
	if (positiv) {
		folder = "INRIAPerson\\train_64x128_H96\\pos";
	}
	else {
		folder = "INRIAPERSON\\Train\\neg";
	}

	//filename depends on i
	string filename;
	//TO-DO something link this is needed:
	//filename = getFilename(i, folder);
	
	

	double*** HoG = extractHOGFeatures(folder, filename, dims);
	double* Template1D = compute1DTemplate(HoG, dims, 0, 0, 0);
	destroy_3Darray(HoG, dims[0], dims[1]);

	return Template1D;
}