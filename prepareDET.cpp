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

float steps = 0.05;
float start = 1.;
float stop = 3.;

void createDETfile() {

	vector<float> pos = testQuantitativDET_pos();
	vector<float> neg = testQuantitativDET_neg();
		
	ofstream DETdata;
	DETdata.open("DETdata_first.txt");
	for (float i = start; i <= stop; i += steps) {
		DETdata << i << endl;
		DETdata << pos[floor((i - start) / steps) + 1] << endl;
		DETdata << neg[floor((i - start) / steps) + 1] << endl;
	}
	DETdata.close(); 

		/*DETdata.open("DETdata_retrained.txt");
		for (float i = min_score; i <= 0; i += 0.1) {
			vector<float> out = testQuantitativDET(i);
			DETdata << i << endl;
			DETdata << out[0] << endl;
			DETdata << out[1] << endl;
		}
		DETdata.close();*/
}

vector<float> testQuantitativDET_neg() {
	cout << "Start testDET_neg" << endl;
	string line;
	ifstream list("INRIAPerson\\Test\\neg.lst");
	int array_size = floor((stop - start) / steps) + 1;
	vector<float> fp(array_size);
	vector<float> fppw(array_size);
	int window_count = 0;

	int counter = 0;

	cout << "Reading in negativ Test Data" << endl;
	while (getline(list, line)) {

		counter++;

		string folder = "INRIAPerson";
		string in = folder + "/" + line;
		int nr_of_templates = 0;
		int* nr_of_templates_ptr = &nr_of_templates;
		vector<templatePos> posTemplates = multiscaleImg(in, nr_of_templates_ptr, start);
		window_count += nr_of_templates;

		vector<templatePos> allTemplates = reduceTemplatesFound(posTemplates, false, in);

		//count false-positives
		for (float i = start; i <= stop; i += steps) {
			for (vector<templatePos>::const_iterator j = allTemplates.begin(); j != allTemplates.end(); ++j) {
				if ((*j).score > (float)i) {
					fp[floor((i - start) / steps) + 1]++;
				}
			}
		}

		if (counter % 5 == 0) {
			cout << "|";
		}
	}
	cout << endl;
	for (float i = start; i <= stop; i += steps) {
		fppw[floor((i - start) / steps) + 1] = fp[floor((i - start) / steps) + 1] / (double)window_count;
		cout << "at i=" << i << " fppw=" << fppw[floor((i - start) / steps) + 1] << endl;
	}

	list.close();
	return fppw;
}

vector<float> testQuantitativDET_pos() {

	string line;
	ifstream list("INRIAPerson\\Test\\pos.lst");
	int array_size = floor((stop - start) / steps) + 1;
	vector<float> misses(array_size);
	vector<float> missrate_total(array_size);
	int bb_count = 0;
	int counter = 0;

	cout << "Reading in positiv Test Data" << endl;
	while (getline(list, line)&& counter < 1000) {
		counter++;

		string folder = "INRIAPerson";
		string in = folder + "/" + line;
		cout << in << endl;
		int nr_of_templates = 0;
		int* nr_of_templates_ptr = &nr_of_templates;
		vector<templatePos> posTemplates = multiscaleImg(in, nr_of_templates_ptr, start);
		cout << posTemplates.size() << endl;
		////vector<templatePos> allTemplates = reduceTemplatesFound(posTemplates, false, in);	
		vector<int> boundingBoxes = getBoundingBoxes(in);

		//cout << "posTemplates.size()=" << posTemplates.size() << "\tallTemplates.size()="<< allTemplates.size() << endl;
		/*for (vector<int>::size_type j = 0; j != allTemplates.size(); j++) {
			cout << "\t\tx=" << allTemplates.at(j).x << " y=" << allTemplates.at(j).y << endl;
		}*/

		for (int k = 0; k < boundingBoxes.size() / 4; k++) {
			for (float i = start; i <= stop; i += steps) {
				cout << i << " ";
				///float out = isFound(allTemplates, boundingBoxes, k, i);
				///if (out < OVERLAP_CORRECT) {
					misses[floor((i - start) / steps) + 1]++;				
				///}
				cout << "misses[" << i << "] = " << misses[floor((i - start) / steps) + 1] << endl;
			}
		}

		bb_count += boundingBoxes.size() / 4;
		cout << "bb_count = " << bb_count << endl;
		if (counter % 5) {
			cout << "|";
		}

		for (float i = start; i <= stop; i += steps) {
			//cout << "at i=" << i << " misses=" << misses[floor((i - start) / steps) + 1] << " with " << bb_count << " boundingboxes" << endl;
		}
		cout << endl;
		//getchar();
	}

	cout << endl;
	for (float i = start; i <= stop; i += steps) {
		missrate_total[floor((i - start) / steps) + 1] = misses[floor((i - start) / steps) + 1] / (double)bb_count;
		cout << "at i=" << i << " missrate=" << missrate_total[floor((i - start) / steps) + 1] << endl;
	}

	list.close();
	
	return missrate_total;
		
}