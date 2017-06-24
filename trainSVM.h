#ifndef TRAINSVM_H_INCLUDED
#define TRAINSVM_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void firstStepTrain();
cv::Mat createFirstSet(int N_pos, int N_neg);
cv::Mat createFirstLabels(int N_pos, int N_neg);
float* getTemplate(std::string filename, bool positiv);

#endif