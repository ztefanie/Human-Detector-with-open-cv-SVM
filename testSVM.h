#ifndef TESTSSVM_H_INCLUDED
#define TESTSSVM_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string.h>
#include "featureExtraction.h"

void testQualitativ();
std::vector<templatePos> multiscaleImg(std::string file);
void reduceTemplatesFound(std::vector<templatePos> posTemplates, bool showOutput, std::string file);
float getOverlap(std::vector<int> truth, cv::Point p1, cv::Point p2);

#endif
