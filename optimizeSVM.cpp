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


Mat find_hardPositives() {

	if (!std::ifstream(SVM_LOCATION)){
		SVMtrain(false);
	}

	CvSVM SVM;
	SVM.load(SVM_LOCATION);

	//Init Mat for output
	Mat allHardPos(0, TEMPLATE_WIDTH_CELLS*TEMPLATE_HEIGHT_CELLS*HOG_DEPTH, CV_32FC1);
	Mat predictMat(0, 1, CV_32FC1);
	cout << endl << "Searching for hard negatives ... " << endl;

	string line;
	ifstream file_list("INRIAPerson\\Train\\pos.lst");
	int i = 0;

	//Iterate over all files
	while (getline(file_list, line)) {
		int count = 0;
		String full_path = "INRIAPerson\\" + line;
		Mat img = imread(full_path);

		double scale = pow(2.0, 1.0 / LAMBDA);
		double akt_width = img.cols;
		double akt_height = img.rows;
		int int_akt_height = floor(akt_height);
		int int_akt_width = floor(akt_width);
		double hig_scale = 1;

		//iterate over sizes
		while (floor(akt_width) >= TEMPLATE_WIDTH && floor(akt_height) >= TEMPLATE_HEIGHT) {
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

			vector<Point> temp_pos;
			vector<Point> real_temp_pos;
			vector<Point> real_temp_size;
			int counter = 0;
			double calc_size = (int_akt_height * int_akt_width * 2) / (TEMPLATE_WIDTH * TEMPLATE_HEIGHT);
			temp_pos.resize(calc_size);
			real_temp_pos.resize(calc_size);
			real_temp_size.resize(calc_size);
			Mat m2 = img.clone();
			for (int i = 0; i + TEMPLATE_HEIGHT <= int_akt_height-16; i += floor(TEMPLATE_HEIGHT / 2)) {
				for (int j = 0; j + TEMPLATE_WIDTH <= int_akt_width -16; j += floor(TEMPLATE_WIDTH / 2)) {

					vector<int> boxes = getBoundingBoxes("INRIAPerson\\" + line);
					int pos = 0;
					double overlap = 0;
					std::vector<int> detected = std::vector<int>(4, 0);
					while (boxes.size() - pos > 3)
					{

						std::vector<int> truth_i = std::vector<int>(4, 0);
						truth_i.at(0) = boxes.at(0 + pos);
						truth_i.at(1) = boxes.at(1 + pos);
						truth_i.at(2) = boxes.at(2 + pos);
						truth_i.at(3) = boxes.at(3 + pos);

						
						detected.at(0) = j * hig_scale;
						detected.at(1) = i * hig_scale;
						detected.at(2) = TEMPLATE_WIDTH * hig_scale + j * hig_scale;
						detected.at(3) = TEMPLATE_HEIGHT * hig_scale + i * hig_scale;

						double overlap_tmp = ComputeOverlap(truth_i, detected);

						pos += 4;
						if (overlap_tmp > overlap) {
							overlap = overlap_tmp;
						}
					}

					if (overlap < 0.4) {

						float* featureTemplate = compute1DTemplate(hog, dims, j/CELL_SIZE, i/CELL_SIZE, 0);
						Mat sampleTest(1, (TEMPLATE_WIDTH_CELLS)*(TEMPLATE_HEIGHT_CELLS)*HOG_DEPTH, CV_32FC1);
						Mat samplePredict(1, 1, CV_32FC1);
						//copy values of template to Matrix
						for (int j = 0; j < sampleTest.cols; j++) {
							sampleTest.at<float>(0, j) = featureTemplate[j];
						}

						samplePredict.at<float>(0, 0) = SVM.predict(sampleTest, true);

						if (samplePredict.at<float>(0, 0) > 0) {

							allHardPos.push_back(sampleTest);
							predictMat.push_back(samplePredict);
						}
						free(featureTemplate);
					}

				}
			}

			count++;
			hig_scale *= scale;

			//destroy at end of each scale
			destroy_3Darray(hog, dims[0], dims[1]);
		}

		if (i % 25 == 0) {
			cout << "|";
		}
		
		i++;
	}

	file_list.close();
	cout << endl << "finished searching for hard negatives found: " << allHardPos.rows << endl << endl;

	//Get only the hard negatives with the most positiv predict
	cout << "Reducing hard negatives to " << MAX_HARD_NEG << "..." << endl;
	Mat out(0, TEMPLATE_WIDTH_CELLS*TEMPLATE_HEIGHT_CELLS*HOG_DEPTH, CV_32FC1);

	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;

	for (int i = 0; i< MAX_HARD_NEG && i < allHardPos.rows; i++) {
		minMaxLoc(predictMat, &minVal, &maxVal, &minLoc, &maxLoc);
		predictMat.at<float>(maxLoc.y, 0) = -1;
		out.push_back(allHardPos.row(maxLoc.y));
	}

	cout << "finished reducing hard negatives to " << out.rows << endl << endl;

	return out;
}