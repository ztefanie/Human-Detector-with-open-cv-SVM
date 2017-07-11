#ifndef TESTS_H_INCLUDED
#define TESTS_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string.h>

void testDrawBoundingBox();
void testOverlapBoundingBox();
void testHog();
void testHogSmallTestImg();
cv::Mat visualizeGradOrientations(double*** hog, std::vector<int> &dims);
void test3DTemplate();
void testDownScale();


#endif
