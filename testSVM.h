#ifndef TESTSSVM_H_INCLUDED
#define TESTSSVM_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string.h>
#include "featureExtraction.h"

void testQualitativ();
std::vector<templatePos> multiscaleImg(std::string file, int* nr_of_templates_ptr, float assumed_positiv);
void reduceTemplatesFound(std::vector<templatePos> posTemplates, bool showOutput, std::string file, int* false_positives, float* miss_rate);
float getOverlap(std::vector<int> truth, cv::Point p1, cv::Point p2);
float isFound(std::vector<templatePos> allTemplates, std::vector<int> truth, int which_bounding_box);

#endif
