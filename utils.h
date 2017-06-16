#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string.h>

//declare functions here
cv::Mat showBoundingBox(std::string filename);
std::vector<int> getBoundingBoxes(std::string file);
double ComputeOverlap(std::vector<int> truth, std::vector<int> detected);
bool isOverlapCorrect(double overlap);
double*** extractHOGFeatures(std::string folder, std::string filename, std::vector<int> &dims);

#endif