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

void trainOptimizedSVM(Mat hardNegatives) {
	//Get Sizes of Datasets
	int N_pos = 0;
	std::string line;
	std::ifstream myfile(LIST_POS);
	while (std::getline(myfile, line))
		++N_pos;

	int N_neg = 0;
	std::ifstream myfile2(LIST_NEG);
	while (std::getline(myfile2, line))
		++N_neg;

	Mat points = createFirstSet(N_pos, N_neg);
	Mat labels = createFirstLabels(N_pos, N_neg);
	Mat label_neg(1, 1, CV_32FC1);
	label_neg.at<float>(0, 0) = 1;

	Mat V;
	vconcat(points, hardNegatives, V);

	for (int i = 0; i < hardNegatives.rows; i++) {
		//points.push_back(hardNegatives.at<float>(i));
		labels.push_back(label_neg);
	}
	
	// Train with SVM
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	//params.C = 0.01; //best option according to Dalal and Triggs
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100000, 1e-6);

	cout << "Retraining SVM with " << V.rows << " Datapoints... " << endl;
	CvSVM SVM;
	SVM.train_auto(V, labels, Mat(), Mat(), params);
	SVM.save(SVM_2_LOCATION);
	cout << "finished retraining SVM" << endl << endl;
}


Mat find_hardNegatives() {

	if (!std::ifstream(SVM_LOCATION)){
		firstStepTrain();
	}

	CvSVM SVM;
	SVM.load(SVM_LOCATION);

	//Init Mat for output
	Mat allHardNeg(0, TEMPLATE_WIDTH_CELLS*TEMPLATE_HEIGHT_CELLS*HOG_DEPTH, CV_32FC1);
	Mat predictMat(0, 1, CV_32FC1);
	cout << endl << "Searching for hard negatives ... " << endl;

	string line;
	ifstream file_neg("INRIAPerson\\train_64x128_H96\\neg.lst");
	int i = 0;

	//Iterate over all files
	while (getline(file_neg, line)) {
		//cout << line << endl;
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

			//iterate over width and height
			if (dims.at(0) > TEMPLATE_HEIGHT_CELLS && dims.at(1) > TEMPLATE_HEIGHT_CELLS) {
				for (int i = 0; i + TEMPLATE_HEIGHT_CELLS < dims.at(0); i += floor(TEMPLATE_HEIGHT_CELLS / 2)) {
					for (int j = 0; j + TEMPLATE_WIDTH_CELLS < dims.at(1); j += floor(TEMPLATE_WIDTH_CELLS / 2)) {
							
						//DO TESTING IF FALSE NEGATIVE
						float* featureTemplate = compute1DTemplate(hog, dims, j, i, 0);
						Mat sampleTest(1, (TEMPLATE_WIDTH_CELLS-2)*(TEMPLATE_HEIGHT_CELLS-2)*HOG_DEPTH, CV_32FC1);
						Mat samplePredict(1, 1, CV_32FC1);
						//copy values of template to Matrix
						for (int j = 0; j < sampleTest.cols; j++) {
							sampleTest.at<float>(0, j) = featureTemplate[j];
						}

						samplePredict.at<float>(0, 0) = SVM.predict(sampleTest, true);
						
						//cout << predict <<  endl;
						if (samplePredict.at<float>(0, 0) > 0) {
							//cout << "found false-negative in " << line << " " << predict << endl;
							allHardNeg.push_back(sampleTest);
							predictMat.push_back(samplePredict);
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
			cout << "|";
		}
		
		i++;
	}

	file_neg.close();
	cout << endl << "finished searching for hard negatives found: " << allHardNeg.rows << endl << endl;

	//Get only the hard negatives with the most positiv predict
	cout << "Reducing hard negatives to " << MAX_HARD_NEG << "..." << endl;
	Mat out(0, TEMPLATE_WIDTH_CELLS*TEMPLATE_HEIGHT_CELLS*HOG_DEPTH, CV_32FC1);

	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;

	for (int i = 0; i< MAX_HARD_NEG && i < allHardNeg.rows; i++) {
		minMaxLoc(predictMat, &minVal, &maxVal, &minLoc, &maxLoc);
		//cout << "max val : " << maxVal << " at " << maxLoc.y << endl;
		predictMat.at<float>(maxLoc.y, 0) = -1;
		out.push_back(allHardNeg.row(maxLoc.y));
	}

	cout << "finished reducing hard negatives to " << MAX_HARD_NEG << endl << endl;

	return out;
}