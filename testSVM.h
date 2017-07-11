#ifndef TESTSSVM_H_INCLUDED
#define TESTSSVM_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string.h>
#include "featureExtraction.h"

void testQualitativ();
void testQualitativRand();
std::vector<templatePos> multiscaleImg(std::string file, int* nr_of_templates_ptr, float assumed_positiv);
std::vector<templatePos> reduceTemplatesFound(std::vector<templatePos> posTemplates, bool showOutput, std::string file);
float getOverlap(std::vector<int> truth, cv::Point p1, cv::Point p2);
float isFound(std::vector<templatePos> allTemplates, std::vector<int> truth, int which_bounding_box, float min_score);
bool compareTemplatePos(templatePos pos1, templatePos pos2); 
bool sortXYScale(templatePos pos1, templatePos pos2);
void visualize(std::vector<templatePos> nonOverlappingTemplates, std::string file);

#endif
