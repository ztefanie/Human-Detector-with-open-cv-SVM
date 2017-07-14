#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include <ctime>

#include "utils.h"
#include "hog.h"
#include "main.h"
#include "featureExtraction.h"

using namespace std;
using namespace cv;

/*
* Comparison Function for templatePos-Struct to order templatePos depending on their score
*
* @returns: true if p1 is smaller, false otherwise
* @param p1: first templatePos which should be compared
* @param p2: second template Pos which should be compared
*
*/
bool compareByScore(templatePos p1, templatePos p2) {
	return p1.score < p2.score;
}

/*
* Function to release allocated space of a 3D array
*
* @param inputArray: array that should be freed
* @param width: width of the array
* @param height: height of the array
*
*/
void destroy_3Darray(double*** inputArray, int width, int height)
{
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			delete[] inputArray[i][j];
		}
		delete[] inputArray[i];
	}
	delete[] inputArray;
}


/* Task 1.3
*
* Function to compute a HoG for a given file
* 
* @returns: HoG-array
* @param folder: folder of the source
* @param filename: filename of the folder
* @dims: dimension of the HoG
*
*/
double*** extractHOGFeatures(string folder, string filename, std::vector<int>& dims)
{
	String get = folder + "\\" + filename;
	Mat img = imread(get, 1);
	return computeHoG(img, CELL_SIZE, dims);
}


/* Task 1.2
*
* Method to compute the overlap for two boxes with the given formula
*
* @returns: calculated overlap
* @param truth: first box - normally the truth-box from the annotations
* @param detected: second box - normally the detected one
*
*/
double ComputeOverlap(std::vector<int> truth, std::vector<int> detected)
{
	int intersect;
	int union1;

	//compute intersect
	int width_intersect;
	int height_intersect;
	if (truth.at(0) < detected.at(0))
	{
		width_intersect = truth.at(2) - detected.at(0);
	}
	else
	{
		width_intersect = detected.at(2) - truth.at(0);
	}
	if (truth.at(1) < detected.at(1))
	{
		height_intersect = truth.at(3) - detected.at(1);
	}
	else
	{
		height_intersect = detected.at(3) - truth.at(1);
	}
	width_intersect < 0 ? width_intersect = 0 : 0;
	height_intersect < 0 ? height_intersect = 0 : 0;

	intersect = width_intersect * height_intersect;

	//compute union
	int size_truth = (truth.at(0) - truth.at(2)) * (truth.at(1) - truth.at(3));
	int size_detected = (detected.at(0) - detected.at(2)) * (detected.at(1) - detected.at(3));
	union1 = size_truth + size_detected - intersect;

	//compute overlap
	double overlap = (double)intersect / union1;
	return overlap;
}

/* Task 1.2
*
* Method to evaluate if a overlap is correct
*
* @returns: if calculated overlap is greater than 50%
* @param overlap: size of the overlap
*
*/
bool isOverlapCorrect(double overlap)
{
	if (overlap > 0.5)
	{
		return true;
	}
	else
	{
		return false;
	}
}


/*
* Read Picture, draw boundingBox inside and show
*
* @returns: image with bounding box drawn inside
* @param img: picture to draw the bounding box inside
* @param file: file where to get the annotations from
*
*/
Mat showBoundingBox(Mat img, string file)
{
std:vector<int> boxes = getBoundingBoxes(file);
	int pos = 0;
	Mat imgClone = img.clone();
	while (boxes.size() - pos > 3)
	{
		Scalar color = Scalar(0, 255, 0);
		Point p1 = Point(boxes.at(pos + 0), boxes.at(pos + 1));
		Point p2 = Point(boxes.at(pos + 0), boxes.at(pos + 3));
		Point p3 = Point(boxes.at(pos + 2), boxes.at(pos + 1));
		Point p4 = Point(boxes.at(pos + 2), boxes.at(pos + 3));
		line(img, p1, p2, color, 0.15);
		line(img, p1, p3, color, 0.15);
		line(img, p2, p4, color, 0.15);
		line(img, p3, p4, color, 0.15);

		pos += 4;
	}
	return img;
}

/*
* Read and parse annotation file and get BoundingBoxes
*
* @returns: vector which contains the data of the BoundingBoxes (xmin, ymin, xmax, ymax) for each box
* @file: filepath of the picture
*
*/
std::vector<int> getBoundingBoxes(string file)
{
	//Change from given picture filepath to annoation filepath
	string line;
	size_t found = file.find("pos");
	file.erase(found, 3);
	file.insert(found, "annotations");
	found = file.find("png");
	file.erase(found, 3);
	file.insert(found, "txt");

	//read file
	ifstream myfile(file);

	std::vector<int> out;
	int pos = 0;
	int size = 0;

	if (myfile.is_open())
	{
		//iterate over lines and parse
		while (getline(myfile, line))
		{
			if (line.find("\"PASperson\" (Xmin, Ymin) - (Xmax, Ymax) : ") != string::npos)
			{
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

/*
* Get Time and Date for naming log-files
*
* returns date-time-string
*
*/
string getTimeLog() {
	time_t t = time(0);   // get time now
	struct tm * now = localtime(&t);
	stringstream ss;
	ss << (now->tm_year + 1900) << '-'
		<< (now->tm_mon + 1) << '-'
		<< now->tm_mday << "-"
		<< now->tm_hour << "-"
		<< now->tm_min;
	string out = ss.str();
	return out;
}

/*
* Get Time and Date for output
*
* returns: nicely formatted Date-Time-String
*/
string getTimeFormatted() {
	time_t t = time(0); 
	struct tm * now = localtime(&t);
	stringstream ss;
	ss << "Date: "
		<< now->tm_mday << '.'
		<< (now->tm_mon + 1) << '.'
		<< (now->tm_year + 1900) << " Time: "
		<< now->tm_hour << ":"
		<< now->tm_min;
	string out = ss.str();
	return out;
}