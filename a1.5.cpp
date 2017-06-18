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

	int tmpl_width = 80;
	int tmpl_height = 160;
	double akt_width = img.cols;
	double akt_height = img.rows;
	Mat oldM = img.clone();
	while (akt_width >= tmpl_width && akt_width >= tmpl_width) {
		if (count % l == 0) {
			double help = pow(2, count / l);
			akt_width = img.cols / help;
			akt_height = img.rows / help;
			Mat m(floor(akt_height), floor(akt_width), CV_8UC3, Scalar(0, 0, 0));
			for (int i = 0; i < floor(akt_height); i++) {
				for (int j = 0; j < floor(akt_width) - 0; j++) {
					m.at<Vec3b>(i, j)[0] = img.at<Vec3b>(i * help, j * help)[0];
					m.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i * help, j * help)[1];
					m.at<Vec3b>(i, j)[2] = img.at<Vec3b>(i * help, j * help)[2];
				}
			}
			imshow("smaler", m);
			waitKey();
			destroyAllWindows();
			oldM = m.clone();
		}
		else {
			akt_width = akt_width / scale;
			akt_height = akt_height / scale;
			Mat m(floor(akt_height), floor(akt_width), CV_8UC3, Scalar(0, 0, 0));
			for (int i = 0; i < floor(akt_height); i++) {
				for (int j = 0; j < floor(akt_width) - 0; j++) {
					m.at<Vec3b>(i, j)[0] = oldM.at<Vec3b>(i * scale, j * scale)[0];
					m.at<Vec3b>(i, j)[1] = oldM.at<Vec3b>(i * scale, j * scale)[1];
					m.at<Vec3b>(i, j)[2] = oldM.at<Vec3b>(i * scale, j * scale)[2];
				}
			}
			imshow("smaler", m);
			waitKey();
			destroyAllWindows();
			oldM = m.clone();
		}
		count++;
	}
}