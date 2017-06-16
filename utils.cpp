#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>

#include "utils.h"

using namespace std;
using namespace cv;

//define functions here


//Read Files and BoundingBoxes
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