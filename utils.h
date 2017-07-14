#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string.h>
#include "featureExtraction.h"


cv::Mat showBoundingBox(cv::Mat img, std::string file);
std::vector<int> getBoundingBoxes(std::string file);
double ComputeOverlap(std::vector<int> truth, std::vector<int> detected);
bool isOverlapCorrect(double overlap);
double*** extractHOGFeatures(std::string folder, std::string filename, std::vector<int>& dims);
void destroy_3Darray(double*** inputArray, int width, int height);
bool compareByScore(templatePos p1, templatePos p2);
std::string getTimeLog();
std::string getTimeFormatted();

#endif
