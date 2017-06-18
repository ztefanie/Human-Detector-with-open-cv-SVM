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

using namespace std;
using namespace cv;

void testDownScale(int l) {
	String file = "INRIAPerson/Train/pos/crop_000607.png";
	Mat img = imread(file);
	int count = 0;

	if (img.empty()) {
		std :: cout << "Error: no Image" << endl;
		system("pause");
		return;
	}

	double scale = pow(2.0, 1.0/l);

	int tmpl_width = 64;
	int tmpl_height = 128;
	double akt_width = img.cols;
	double akt_height = img.rows;
	int int_akt_height = floor(akt_height);
	int int_akt_width = floor(akt_width);
	double hig_scale = 1;
	//Mat oldM = img.clone();
	while (floor(akt_width) >= tmpl_width && floor(akt_height) >= tmpl_height) {
		if (count % l == 0) {
			double help = pow(2, count / l);
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
		double calc_size = (int_akt_height * int_akt_width * 2) / (tmpl_width * tmpl_height);
		temp_pos.resize(calc_size);
		real_temp_pos.resize(calc_size);
		real_temp_size.resize(calc_size);
		Mat m2 = img.clone(); //just for viso
		for (int i = 0; i + tmpl_height <= int_akt_height; i+= floor(tmpl_height/2)) {
			for (int j = 0; j + tmpl_width <= int_akt_width; j+= floor(tmpl_width/2)) {
				temp_pos[counter] = Point(j, i);
				real_temp_pos[counter] = Point(j * hig_scale, i * hig_scale);
				real_temp_size[counter] = Point(tmpl_width * hig_scale + j * hig_scale, tmpl_height * hig_scale + i * hig_scale);
				//just for viso:
				rectangle(m, temp_pos[counter], Point(j + tmpl_width, i + tmpl_height), CV_RGB(255, 255, 0), 1, 8);
				rectangle(m2, real_temp_pos[counter], real_temp_size[counter], CV_RGB(255, 255, 0), 1, 8);
			}
		}
		//just for viso
		rectangle(m, Point(0, 0), Point(tmpl_width, tmpl_height), CV_RGB(0, 0, 255), 1, 8);
		rectangle(m2, Point(0, 0), Point(tmpl_width * hig_scale, tmpl_height * hig_scale), CV_RGB(0, 0, 255), 1, 8);
		imshow("ori", m2);
		imshow("smaler", m);
		waitKey();
		destroyAllWindows();
		count++;
		hig_scale *= scale;
	}
}