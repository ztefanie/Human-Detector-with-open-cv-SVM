#ifndef TRAINSVM_H_INCLUDED
#define TRAINSVM_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void firstStepTrain();
Mat createFirstSet(int N);
Mat createFirstLabels(int N);
double* getTemplate(int i, bool positiv);

#endif