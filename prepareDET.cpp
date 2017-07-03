#define _USE_MATH_DEFINES

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>

#include "prepareDET.h"
#include "DET.h"
#include "tests.h"
#include "utils.h"
#include "hog.h"
#include "main.h"
#include "optimizeSVM.h"
#include "testSVM.h"

using namespace std;
using namespace cv;

void createDETfile() {

	ofstream DETdata;
	DETdata.open("DETdata_first.txt");
	for (float i = -1; i <= 1; i += 0.5) {
		vector<float> out = testQuantitativDET(i);
		DETdata << i << endl;
		DETdata << out[0] << endl;
		DETdata << out[1] << endl;
	}
	DETdata.close();


	/*DETdata.open("DETdata_retrained.txt");
	for (float i = -2; i <= 0; i += 0.1) {
		vector<float> out = testQuantitativ(i);
		DETdata << i << endl;
		DETdata << out[0] << endl;
		DETdata << out[1] << endl;
	}
	DETdata.close();*/
}

vector<float> testQuantitativDET(float assumed_positiv) {

	string line;
	ifstream list_pos("INRIAPerson\\Test\\pos.lst");
	int test_size = 0;
	double miss_rate_total = 0;
	double fppw_total = 0;

	cout << "Reading in Test Data" << endl;
	while (getline(list_pos, line) && test_size < 25) {
		test_size++;
		string folder = "INRIAPerson";
		string in = folder + "/" + line;
		//cout << in;
		int nr_of_templates = 0;
		int* nr_of_templates_ptr = &nr_of_templates;
		int false_pos = 0;
		int* false_pos_ptr = &false_pos;
		float miss_rate = 0;
		float* miss_rate_ptr = &miss_rate;
		vector<templatePos> posTemplates = multiscaleImg(in, nr_of_templates_ptr, assumed_positiv);
		reduceTemplatesFound(posTemplates, false, in, false_pos_ptr, miss_rate_ptr);
		miss_rate_total += miss_rate;
		fppw_total += false_pos / (double)nr_of_templates;
		//cout << " " << miss_rate << " " << false_pos << endl;
		cout << "|" ;
	}
	list_pos.close();

	vector<float> out = { 0.,0. };
	out.at(0) = fppw_total / test_size;
	out.at(1) = miss_rate_total / test_size;

	cout << endl << "At " << assumed_positiv << "\t\tmiss rate total = " << out.at(1) << " FPPW total = " << out.at(0) << endl << endl;

	return out;
}