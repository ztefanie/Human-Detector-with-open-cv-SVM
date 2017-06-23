#ifndef TRAINSVM_H_INCLUDED
#define TRAINSVM_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void firstStepTrain();
cv::Mat createFirstSet(int N);
cv::Mat createFirstLabels(int N);
float* getTemplate(std::string filename, bool positiv);

#endif