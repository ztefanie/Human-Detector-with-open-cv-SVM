#ifndef FEATUREEXTRACTION_H_INCLUDED
#define FEATUREEXTRACTION_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef struct {
	int x;
	int y;
	float scale;
	float score;
}templatePos;

double*** compute3DTemplate(double*** hog, const std::vector<int> &dims, int grid_pos_x, int grid_pos_y);
float* compute1DTemplate(double*** hog, const std::vector<int> &dims, int grid_pos_x, int grid_pos_y, int scale);
std::vector<templatePos> multiscaleImg(std::string file);
void reduceTemplatesFound(std::vector<templatePos> posTemplates, bool showOutput);
float getOverlap(std::vector<int> truth, cv::Point p1, cv::Point p2);


#endif