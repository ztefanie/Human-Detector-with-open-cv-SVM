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

/*void trainOptimizedSVM(Mat hardNegatives) {
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
}*/


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
		//cout << full_path << endl;
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
			//Mat hogMat = visualizeGradOrientations(hog, dims);
			//imshow("HogMat", hogMat);

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
					//temp_pos[counter] = Point(j, i);
					//real_temp_pos[counter] = Point(j * hig_scale, i * hig_scale);
					//real_temp_size[counter] = Point(TEMPLATE_WIDTH * hig_scale + j * hig_scale, TEMPLATE_HEIGHT * hig_scale + i * hig_scale);
					//rectangle(m, temp_pos[counter], Point(j + TEMPLATE_WIDTH, i + TEMPLATE_HEIGHT), CV_RGB(255, 255, 0), 1, 8);
					//rectangle(m2, real_temp_pos[counter], real_temp_size[counter], CV_RGB(255, 255, 0), 1, 8);

					vector<int> boxes = getBoundingBoxes("INRIAPerson\\" + line);
					int pos = 0;
					double overlap = 0;
					std::vector<int> detected = std::vector<int>(4, 0);
					while (boxes.size() - pos > 3)
					{
						//Mat img = imread("INRIAPerson\\" + line);

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

						//cout << detected.at(0) << ", " << detected.at(1) << " overlap: " << overlap_tmp << endl;

						pos += 4;
						if (overlap_tmp > overlap) {
							overlap = overlap_tmp;
						}
					}

					if (overlap < 0.4) {
						//DO TESTING IF FALSE POSITIV
						/*double*** featureTemplate2 = compute3DTemplate(hog, dims, j/CELL_SIZE, i/CELL_SIZE);
						vector<int> dims2 = vector<int>(3);
						dims2[0] = TEMPLATE_HEIGHT_CELLS;
						dims2[1] = TEMPLATE_WIDTH_CELLS;
						dims2[2] = HOG_DEPTH;
						//Mat out2 = visualizeGradOrientations(featureTemplate2, dims2);
						//imshow("Hog of template at given poistion", out2);*/

						float* featureTemplate = compute1DTemplate(hog, dims, j/CELL_SIZE, i/CELL_SIZE, 0);
						Mat sampleTest(1, (TEMPLATE_WIDTH_CELLS)*(TEMPLATE_HEIGHT_CELLS)*HOG_DEPTH, CV_32FC1);
						Mat samplePredict(1, 1, CV_32FC1);
						//copy values of template to Matrix
						for (int j = 0; j < sampleTest.cols; j++) {
							sampleTest.at<float>(0, j) = featureTemplate[j];
						}

						samplePredict.at<float>(0, 0) = SVM.predict(sampleTest, true);

						//cout << "overlap = " << overlap << "score = " << samplePredict.at<float>(0, 0) << endl;

						//cout << predict <<  endl;
						if (samplePredict.at<float>(0, 0) > 0) {
							//cout << "found false-negative in " << line << " " << predict << endl;
							/*int baseline = 0;
							int size = getTextSize("blubb", CV_FONT_HERSHEY_SIMPLEX, TEMPLATE_WIDTH * hig_scale / 400, 1, &baseline).height;
							rectangle(img, Point(detected.at(0), detected.at(1)), Point(detected.at(2), detected.at(3)), Scalar(255, 0, 0));
							String selection_score = "Selection Score: " + to_string(samplePredict.at<float>(0, 0));
							putText(img, selection_score, Point(detected.at(0) + 2, detected.at(1) + size + 2), CV_FONT_HERSHEY_SIMPLEX, TEMPLATE_WIDTH * hig_scale / 500, cvScalar(0, 255, 0), 1, CV_AA);
							String overlapS = "Overlap: " + to_string(overlap);
							putText(img, overlapS, Point(detected.at(0) +2 , detected.at(1) + 6), CV_FONT_HERSHEY_SIMPLEX, TEMPLATE_WIDTH * hig_scale / 500, cvScalar(0, 255, 0), 1, CV_AA);
							*/
							//rectangle(img, Point(boxes.at(0 + pos), boxes.at(1 + pos)), Point(boxes.at(2 + pos), boxes.at(3 + pos)), Scalar(0, 255, 50));

							allHardPos.push_back(sampleTest);
							predictMat.push_back(samplePredict);
						}
						free(featureTemplate);
					}

					//imshow("m2", m2);
					//waitKey();
				}
			}

			count++;
			hig_scale *= scale;

			//destroy at end of each scale
			destroy_3Darray(hog, dims[0], dims[1]);
		}

		//imshow("Bild", img);
		//waitKey();
		//////////
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
		//cout << "max val : " << maxVal << " at " << maxLoc.y << endl;
		predictMat.at<float>(maxLoc.y, 0) = -1;
		out.push_back(allHardPos.row(maxLoc.y));
	}

	cout << "finished reducing hard negatives to " << out.rows << endl << endl;

	return out;
}