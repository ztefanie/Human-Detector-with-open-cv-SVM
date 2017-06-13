#ifndef __Functions_H_INCLUDED__
#define __Functions_H_INCLUDED__

//include dependencies
#include <algorithm>
#include <ctime>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <math.h>
#define _USE_MATH_DEFINES	//Make use of mathematical constants
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>
#include <random>
#include <string>
#include <sstream>
#include <vector>
//--------------------------------

//set namespaces
using namespace std;
using namespace cv;
//--------------------------------

//Functions
//Sonstiges
#ifdef _WIN32
#include <windows.h>
void colorConsole(int color);
#endif
string convertStringToLower(string str);
int newRandomintGenerator(int min, int max);
double newRandomdoubleGenerator(double min, double max);

//Blatt00 - 0.3
int RandomintGenerator(int MAX, int MIN);
double RandomdoubleGenerator(double MAX, double MIN);
int WriteText(string path, int n, string text);
//--------------------------------

//Blatt00 - 0.4
bool file_exists(string fn);
string compare(string Text, string Variable);
string doubleTostring(double val);
string intTostring(int val);
double caclvarianz(double i, double n);
double CalcMHWScore(vector<int> scores);
//--------------------------------

//Blatt01 - 1.1
void createHistogram(Mat img, int b, string name);
void enhanceBrightnessAndContrast(Mat image, int b, string name);
//--------------------------------

//Blatt02 - 2.1
Mat createPattern(int size);
double getValue(int a, int b);
Mat filterImg(Mat img, int kernel, int filter);
uchar boxFilter(Mat img, int kernel, int x, int y);
uchar gaussianFilter(Mat img, int kernel, int x, int y);
uchar medianFilter(Mat img, int kernel, int x, int y);
Mat combineFourImg(Mat img1, Mat img2, Mat img3, Mat img4);
//--------------------------------

//Blatt02 - 2.2
Mat filterImg(Mat img, int direction);
int Sobel(Mat img, int x, int y, int dir);
Mat Viszualize(Mat img, Mat magnitude, Mat directions);
//--------------------------------

//Blatt03 - 3.1
double*** compute_HoG(const cv::Mat& img, const int cell_size, std::vector<int>& dims);
Mat gradients(Mat img);
//--------------------------------

//Blatt03 - 3.2
Mat visualize_HoG(double*** hog, vector<int>& dims);
//--------------------------------

//Blatt04 - 1
Mat createSimpleTrainingSet(int n, int cols, int rows, int spare, Mat trainingDataMat, float* labels);
Mat drawSet(Mat vals);
Mat trainSVM(int n, int width, int height, int spare, int kernel);
Mat testSVM(int width, int height);
#endif // __Functions_H_INCLUDED__
