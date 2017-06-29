#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include <math.h>

#include "tests.h"
#include "utils.h"
#include "hog.h"
#include "main.h"

using namespace std;
using namespace cv;

void testDownScale() {
	String file = "INRIAPerson/Train/pos/crop_000607.png";
	Mat img = imread(file);
	int count = 0;

	if (img.empty()) {
		std :: cout << "Error: no Image" << endl;
		system("pause");
		return;
	}

	double scale = pow(2.0, 1.0/LAMBDA);

	double akt_width = img.cols;
	double akt_height = img.rows;
	int int_akt_height = floor(akt_height);
	int int_akt_width = floor(akt_width);
	double hig_scale = 1;
	//Mat oldM = img.clone();
	while (floor(akt_width) >= TEMPLATE_WIDTH && floor(akt_height) >= TEMPLATE_HEIGHT) {
		if (count % LAMBDA == 0) {
			double help = pow(2, count / LAMBDA);
			akt_width = img.cols / help;
			akt_height = img.rows / help;
			/*for (int i = 0; i < floor(akt_height); i++) {
				for (int j = 0; j < floor(akt_width); j++) {
					m.at<Vec3b>(i, j)[0] = img.at<Vec3b>(i * help, j * help)[0];
					m.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i * help, j * help)[1];
					m.at<Vec3b>(i, j)[2] = img.at<Vec3b>(i * help, j * help)[2];
				}
			}*/
			//oldM = m.clone();
		}
		else {
			akt_width = akt_width / scale;
			akt_height = akt_height / scale;
			/*for (int i = 0; i < floor(akt_height); i++) {
				for (int j = 0; j < floor(akt_width); j++) {
					m.at<Vec3b>(i, j)[0] = oldM.at<Vec3b>(i * scale, j * scale)[0];
					m.at<Vec3b>(i, j)[1] = oldM.at<Vec3b>(i * scale, j * scale)[1];
					m.at<Vec3b>(i, j)[2] = oldM.at<Vec3b>(i * scale, j * scale)[2];
				}
			}*/
			//oldM = m.clone();
		}
		int_akt_height = floor(akt_height);
		int_akt_width = floor(akt_width);
		Mat m(int_akt_height, int_akt_width, CV_8UC3, Scalar(0, 0, 0));
		resize(img, m, Size(int_akt_width, int_akt_height));
		cout << int_akt_height << " " << int_akt_width << endl;
		//erstmal 50% überlappung; bei 75% durch 4 teilen:
		vector<Point> temp_pos;
		vector<Point> real_temp_pos;
		vector<Point> real_temp_size;
		int counter = 0;
		double calc_size = (int_akt_height * int_akt_width * 2) / (TEMPLATE_WIDTH * TEMPLATE_HEIGHT);
		temp_pos.resize(calc_size);
		real_temp_pos.resize(calc_size);
		real_temp_size.resize(calc_size);
		Mat m2 = img.clone(); //just for viso
		for (int i = 0; i + TEMPLATE_HEIGHT <= int_akt_height; i+= floor(TEMPLATE_HEIGHT/2)) {
			for (int j = 0; j + TEMPLATE_WIDTH <= int_akt_width; j+= floor(TEMPLATE_WIDTH/2)) {
				temp_pos[counter] = Point(j, i);
				real_temp_pos[counter] = Point(j * hig_scale, i * hig_scale);
				real_temp_size[counter] = Point(TEMPLATE_WIDTH * hig_scale + j * hig_scale, TEMPLATE_HEIGHT * hig_scale + i * hig_scale);
				//just for viso:
				rectangle(m, temp_pos[counter], Point(j + TEMPLATE_WIDTH, i + TEMPLATE_HEIGHT), CV_RGB(255, 255, 0), 1, 8);
				rectangle(m2, real_temp_pos[counter], real_temp_size[counter], CV_RGB(255, 255, 0), 1, 8);
			}
		}
		//just for viso
		rectangle(m, Point(0, 0), Point(TEMPLATE_WIDTH, TEMPLATE_HEIGHT), CV_RGB(0, 0, 255), 1, 8);
		rectangle(m2, Point(0, 0), Point(TEMPLATE_WIDTH * hig_scale, TEMPLATE_HEIGHT * hig_scale), CV_RGB(0, 0, 255), 1, 8);
		imshow("ori", m2);
		imshow("smaler", m);
		waitKey();
		destroyAllWindows();
		count++;
		hig_scale *= scale;
	}
}