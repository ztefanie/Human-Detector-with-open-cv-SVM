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

#include "optimizeSVM.h"
#include "main.h"

using namespace std;
using namespace cv;

Mat find_hardNegatives() {


	if (!std::ifstream(SVM_LOCATION)){
		firstStepTrain();
	}

	CvSVM SVM;
	SVM.load(SVM_LOCATION);

	//Init Mat for output
	Mat out(0, TEMPLATE_WIDTH_CELLS*TEMPLATE_HEIGHT_CELLS, CV_32FC1);
	cout << endl << "Searching for hard negatives ... " << endl;

	string line;
	ifstream file_neg("INRIAPerson\\train\\neg.lst");
	int i = 0;

	//Iterate over all files
	while (getline(file_neg, line)) {
		cout << line << endl;
		int count = 0;
		String full_path = "INRIAPerson\\" + line;
		Mat img = imread(full_path);

		double scale = pow(2.0, 1.0 / LAMBDA);
		double akt_width = img.cols;
		double akt_height = img.rows;
		int int_akt_height = floor(akt_height);
		int int_akt_width = floor(akt_width);
		double hig_scale = 1;
		
		//cout << "width = " << akt_width << " height = " << akt_height << endl;

		//iterate over sizes
		while (floor(akt_width) >= TEMPLATE_WIDTH && floor(akt_height) >= TEMPLATE_HEIGHT) {
			//cout << "\t" << count << endl;
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

			//compute HOG for every size
			vector<int> dims;
			double*** hog = computeHoG(m, CELL_SIZE, dims);
			Mat out = visualizeGradOrientations(hog, dims);
			String pic = "Gradients at scale: " + to_string(count);

			//iterate over width and height
			if (dims.at(0) > TEMPLATE_HEIGHT_CELLS && dims.at(1) > TEMPLATE_HEIGHT_CELLS) {
				for (int i = 0; i + TEMPLATE_HEIGHT_CELLS < dims.at(0); i += floor(TEMPLATE_HEIGHT_CELLS / 2)) {
					for (int j = 0; j + TEMPLATE_WIDTH_CELLS < dims.at(1); j += floor(TEMPLATE_WIDTH_CELLS / 2)) {
							
						//DO TESTING IF FALSE NEGATIVE
						float* featureTemplate = compute1DTemplate(hog, dims, j, i, 0);
						Mat sampleTest(1, TEMPLATE_WIDTH_CELLS*TEMPLATE_HEIGHT_CELLS, CV_32FC1);
						//copy values of template to Matrix
						for (int j = 0; j < sampleTest.cols; j++) {
							sampleTest.at<float>(0, j) = featureTemplate[j];
						}

						float predict = SVM.predict(sampleTest, true);
						//cout << predict <<  endl;
						if (predict > 0) {
							cout << "found false-negative in " << line << " " << predict << endl;
							//out.push_back(sampleTest);
						}
					}
				}
			}

			count++;
			hig_scale *= scale;

			//destroy at end of each scale
			destroy_3Darray(hog, dims[0], dims[1]);
		}
		//////////
		if (i % 50 == 0) {
			//cout << "|";
		}
		
		i++;
	}

	cout << endl << "finished searching for hard negatives" << endl << endl;
	file_neg.close();

	return out;
}