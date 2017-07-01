#define _USE_MATH_DEFINES

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>


#include "DET.h"
#include "tests.h"
#include "utils.h"
#include "hog.h"
#include "main.h"
#include "optimizeSVM.h"





using namespace std;
using namespace cv;

void createDET() {

	vector<float> responses_pos_first = createResponse(true, true);
	vector<float> responses_neg_first = createResponse(true, false);

	ofstream DETdata;
	DETdata.open("DETdata_first.txt");
	for (float i = -2; i <= 0; i += 0.1) {
		vector<float> out = testQuantitativ(i, responses_pos_first, responses_neg_first);
		DETdata << i << endl;
		DETdata << out[0] << endl;
		DETdata << out[1] << endl;
	}
	DETdata.close();

	vector<float> responses_pos_retrained = createResponse(false, true);
	vector<float> responses_neg_retrained = createResponse(false, false);
	DETdata.open("DETdata_retrained.txt");
	for (float i = -2; i <= 0; i += 0.1) {
		vector<float> out = testQuantitativ(i, responses_pos_retrained, responses_neg_retrained);
		DETdata << i << endl;
		DETdata << out[0] << endl;
		DETdata << out[1] << endl;
	}
	DETdata.close();
}

vector<float> createResponse(bool first, bool positiv) {
	CvSVM SVM;
	if (first) {
		SVM.load(SVM_LOCATION);
	}
	else {
		SVM.load(SVM_2_LOCATION);
	}

	Mat sampleTest(1, (TEMPLATE_WIDTH_CELLS-2)*(TEMPLATE_HEIGHT_CELLS-2)*HOG_DEPTH, CV_32FC1);
	int testSize = 0;
	string line;
	ifstream test_lst;
	if (positiv) {
		test_lst.open("INRIAPerson\\test_64x128_H96\\pos.lst");
		cout << "positiv Images are tested ... " << endl;
	}
	else {
		test_lst.open("INRIAPerson\\Test\\neg.lst");
		cout << "negativ Images are tested ... " << endl;
	}
	while (getline(test_lst, line))
		++testSize;

	test_lst.clear();
	test_lst.seekg(0, ios::beg);

	//Test positiv images
	vector<float> responses;
	float sum_pos = 0;
	int false_negatives = 0;
	
	for (int i = 0; i < testSize; i++) {
		getline(test_lst, line);
		if (positiv) {
			line.insert(4, "_64x128_H96");
		}
		float* templateHoG;
		templateHoG = getTemplate(line, true, false);

		//copy values of template to Matrix
		for (int j = 0; j < sampleTest.cols; j++) {
			sampleTest.at<float>(0, j) = templateHoG[j];
		}

		//check
		float response = SVM.predict(sampleTest, true);
		responses.push_back(response);
	}
	test_lst.close();

	cout << "tested " << responses.size() << " images" << endl;
	return responses;
}

vector<float> testQuantitativ(float assumed_positiv, vector<float>& responses_pos, vector<float>& responses_neg) {

	int false_positives = 0;
	for (int i=0; i < responses_neg.size(); i++) {
		if (responses_neg[i] >= assumed_positiv) {
			false_positives++;
		}
	}

	int false_negatives = 0;
	for (int i=0; i < responses_pos.size(); i++) {
		if (responses_pos[i] < assumed_positiv) {
			false_negatives++;
		}
	}

	cout << "False positives = " << false_positives << " false negatives = " << false_negatives << endl;

	vector<float> out = { 0.,0. };
	//FPPW
	out.at(0) = false_positives / (float)responses_neg.size();
	//miss rate
	out.at(1) = 1 - (((float)responses_pos.size() - false_negatives) / (float)responses_pos.size());

	cout << "miss rate = " << out.at(1) << " FPPW = " << out.at(0) << endl << endl;


	return out;
}