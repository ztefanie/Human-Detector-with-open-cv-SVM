#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>

#include "utils.h"

using namespace std;
using namespace cv;


//compute overlap
double ComputeOverlap(std::vector<int> truth, std::vector<int> detected) {
	int intersect;
	int union1;

	//compute intersect
	int width_intersect;
	int height_intersect;
	if (truth.at(0) < detected.at(0)) {
		width_intersect = truth.at(2) - detected.at(0);
	} else {
		width_intersect = detected.at(2) - truth.at(0);
	}
	if (truth.at(1) < detected.at(1)) {
		height_intersect = truth.at(3) - detected.at(1);
	}
	else {
		height_intersect = detected.at(3) - truth.at(1);
	}
	width_intersect < 0 ? width_intersect = 0 : 0;
	height_intersect < 0 ? height_intersect = 0 : 0;

	intersect = width_intersect * height_intersect;

	//compute union
	int size_truth = (truth.at(0) - truth.at(2))*(truth.at(1) - truth.at(3));
	int size_detected = (detected.at(0) - detected.at(2))*(detected.at(1) - detected.at(3));
	union1 = size_truth + size_detected - intersect;

	//compute overlap
	double overlap = (double)intersect / union1;
	return overlap;
}

//compate
bool isOverlapCorrect(double overlap) {
	if (overlap > 0.5) {
		return true;
	}
	else {
		return false;
	}
}


//Read Picture, draw boundingBox inside and show
Mat showBoundingBox(string file) {
	string get = "INRIAPerson\\Train\\pos\\" + file + ".png";
	Mat img = imread(get, 1);
std:vector<int> boxes = getBoundingBoxes(file);
	int pos = 0;
	while (boxes.size() - pos > 3) {
		Scalar color = Scalar(0, 255, 0);
		Point p1 = Point(boxes.at(pos + 0), boxes.at(pos + 1));
		Point p2 = Point(boxes.at(pos + 0), boxes.at(pos + 3));
		Point p3 = Point(boxes.at(pos + 2), boxes.at(pos + 1));
		Point p4 = Point(boxes.at(pos + 2), boxes.at(pos + 3));
		line(img, p1, p2, color, 5);
		line(img, p1, p3, color, 5);
		line(img, p2, p4, color, 5);
		line(img, p3, p4, color, 5);
		pos += 4;
	}
	return img;
}

//Read annotation file and get BoundingBoxes
std::vector<int> getBoundingBoxes(string file) {
	string line;
	string get = "INRIAPerson\\Train\\annotations\\" + file + ".txt";
	ifstream myfile(get);
	int Xmin, Ymin, Xmax, Ymax;

	std::vector<int> out;
	int pos = 0;
	int size = 0;

	if (myfile.is_open())
	{
		while (getline(myfile, line)) {
			//cout << line << endl;
			if (line.find("\"PASperson\" (Xmin, Ymin) - (Xmax, Ymax) : ") != string::npos) {
				size++;
				out.resize(4 * size);
				//get first int
				size_t start = line.find(") : (");
				string substring = line.substr(start + 5);
				size_t end = substring.find(",");
				out[pos] = stoi(substring.substr(0, end));
				pos++;
				//get second int
				start = substring.find(",");
				substring = substring.substr(start + 2);
				end = substring.find(")");
				out[pos] = stoi(substring.substr(0, end));
				pos++;
				//get third int
				start = substring.find("(");
				substring = substring.substr(start + 1);
				end = substring.find(",");
				out[pos] = stoi(substring.substr(0, end));
				pos++;
				//get fourth int
				start = substring.find(",");
				substring = substring.substr(start + 2);
				end = substring.find(")");
				out[pos] = stoi(substring.substr(0, end));
				pos++;
			}
		}
		myfile.close();
	}

	else cout << "Unable to open file";
	return out;
}