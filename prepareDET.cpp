#define _USE_MATH_DEFINES

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>

#include "prepareDET.h"
#include "tests.h"
#include "utils.h"
#include "hog.h"
#include "main.h"
#include "optimizeSVM.h"
#include "testSVM.h"

using namespace std;
using namespace cv;

float steps = 0.01;
float start = -2.5;
float stop = 1.9;

/* Task 3
*
* creates DETFiles which are used by a python script to create the DET-Plots
*
*/
void createDETfile() {
	cout << "Creating DET-File for first SVM... " << endl;
	vector<float> pos = testQuantitativDET_pos(true);
	vector<long double> neg = testQuantitativDET_neg(true);

	ofstream DETdata;
	DETdata.precision(17);
	DETdata.open("DETdata_first.txt");
	for (float i = start; i <= stop; i += steps) {
		DETdata << i << endl;
		DETdata << fixed << pos[floor((i - start) / steps) + 1] << endl;
		DETdata << fixed << neg[floor((i - start) / steps) + 1] << endl;
	}
	DETdata.close(); 
	cout << "finished creating DET-File for first SVM... " << endl;


	cout << endl << "Creating DET-File for second SVM... " << endl;
	vector<float> pos2 = testQuantitativDET_pos(false);
	vector<long double> neg2 = testQuantitativDET_neg(false);

	ofstream DETdata2;
	DETdata2.precision(17);
	DETdata2.open("DETdata_retrained.txt");
	for (float i = start; i <= stop; i += steps) {
		DETdata2 << i << endl;
		DETdata2 << fixed << pos2[floor((i - start) / steps) + 1] << endl;
		DETdata2 << fixed << neg2[floor((i - start) / steps) + 1] << endl;
	}
	DETdata2.close();
	cout << "finished creating DET-File for second SVM... " << endl;

}

/* Task 3
*
* tests negativ test-Images
* 
* @returns: vector of different false positiv per windows depending on detection score
* @params first: boolean if the first or the second SVM should be tested
*
*/
vector<long double> testQuantitativDET_neg(bool first) {

	CvSVM SVM;
	if (first) {
		SVM.load(SVM_LOCATION);
	}
	else {
		SVM.load(SVM_2_LOCATION);
	}

	string line;
	ifstream list("INRIAPerson\\Test\\neg.lst");

	int factor = 100;
	int test_size = 453 * factor;

	const int template_size = (TEMPLATE_WIDTH_CELLS)*(TEMPLATE_HEIGHT_CELLS)*HOG_DEPTH;
	Mat temp_neg(test_size, template_size, CV_32FC1);
	int array_size = floor((stop - start) / steps) + 1;
	vector<int> fp(array_size);
	vector<long double> fppw(array_size);

	cout << "Reading in negativ test data ..." << endl;
	for (int i = 0; i < test_size; i += factor) {
		getline(list, line);
		//for each negativ 10 templates
		for (int k = 0; k < factor; k++) {
			float* templateHoG;
			templateHoG = getTemplate(line);

			//copy values of template to Matrix
			for (int j = 0; j < temp_neg.cols; j++) {
				temp_neg.at<float>(i + k, j) = templateHoG[j];
			}
		}
		//cout << "point at i=" << i << " from " << line_neg << endl;
		if (i % 400 == 0) {
			cout << "|";
		}
	}
	cout << endl << "Finished reading in negativ test data ..." << endl;
	list.close();
	
	Mat template_temp(1, template_size, CV_32FC1);
	//iterate over rows ot points_neg
	for (int row = 0; row < temp_neg.rows; row++) {
		//fill float-array
		for (int i = 0; i < template_size; i++) {
			template_temp.at<float>(0, i) = temp_neg.at<float>(row, i);
		}
		float score = SVM.predict(template_temp, true);

		//iterate over miss array
		for (float i = start; i <= stop; i += steps) {
			if (score > i) {
				fp[floor((i - start) / steps) + 1]++;
			}
		}
	}

	for (float i = start; i <= stop; i += steps) {
		fppw[floor((i - start) / steps) + 1] = fp[floor((i - start) / steps) + 1] / (long double)test_size;
	}

	return fppw;
}

/* Task 3
*
* tests positiv test-Images
*
* @returns: vector of different miss-rates depending on detection score
* @params first: boolean if the first or the second SVM should be tested
*
*/
vector<float> testQuantitativDET_pos(bool first) {
	CvSVM SVM;
	if (first) {
		SVM.load(SVM_LOCATION);
	}
	else {
		SVM.load(SVM_2_LOCATION);
	}

	const int template_size = (TEMPLATE_WIDTH_CELLS)*(TEMPLATE_HEIGHT_CELLS)*HOG_DEPTH;
	Mat people_pos(1178, template_size, CV_32FC1);

	string line;
	ifstream list("INRIAPerson\\Test\\pos.lst");
	int last = 0;
	int i = 0;
	cout << "Reading in positiv test data ..." << endl;
	while (getline(list, line)) {
		float* templateHoG;
		get1DTemplateFromPos(line, people_pos, &last, false);
		if (i % 10 == 0) {
			cout << "|";
		}
		i++;
	}
	list.close();

	cout << endl << "Finished reading in positiv test data ..." << endl;

	Mat template_temp(1, template_size, CV_32FC1);
	int array_size = floor((stop - start) / steps) + 1;
	vector<float> misses(array_size);
	vector<float> missrate_total(array_size);

	//iterate over rows ot points_temp_pos
	for (int row = 0; row < last; row++) {
		//fill float-array
		for (int i = 0; i < template_size; i++) {
			template_temp.at<float>(0,i) = people_pos.at<float>(row, i);
		}
		float score = SVM.predict(template_temp, true);
		
		//iterate over miss array
		for (float i = start; i <= stop; i += steps) {
			if (score < i) {
				misses[floor((i - start) / steps) + 1]++;
			}
		}
	}

	for (float i = start; i <= stop; i += steps) {
		missrate_total[floor((i - start) / steps) + 1] = misses[floor((i - start) / steps) + 1] / (double)last;
	}		
	return missrate_total;
}