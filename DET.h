#ifndef DET_H_INCLUDED
#define DET_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void createDET();
std::vector<float> createResponse(bool first, bool positiv);
std::vector<float> testQuantitativ(float assumed_positiv, std::vector<float>& responses_pos, std::vector<float>& responses_neg);

#endif