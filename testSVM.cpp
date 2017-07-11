#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <iostream>
#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h> 
#include <stdlib.h>

#include "testSVM.h"
#include "trainSVM.h"
#include "hog.h"
#include "featureExtraction.h"
#include "main.h"
#include "utils.h"
#include "tests.h"


using namespace std;
using namespace cv;

/*
* For presentations. Takes random line of positiv-list and shows output of the SVM
*/
void testQualitativRand() {
	string line;
	ifstream list_pos("INRIAPerson\\Test\\pos.lst");
	int line_ctr = 0;

	while (getline(list_pos, line)) {
		line_ctr++;
	}
	list_pos.clear();
	list_pos.seekg(0, ios::beg);

	cout << "Number of lines: " << line_ctr << endl;

	while (true) {
		//back to first line
		list_pos.clear();
		list_pos.seekg(0, ios::beg);

		//get random line
		int pic_rand = rand() % line_ctr;
		for (int i = 0; i < pic_rand; i++)
		{
			getline(list_pos, line);
		}

		//get picture
		string folder = "INRIAPerson";
		string in = folder + "/" + line;
		cout << in << endl;
		int nr_of_templates = 0;
		int* nr_of_templates_ptr = &nr_of_templates;

		//find persons
		vector<templatePos> posTemplates;
		posTemplates = multiscaleImg(in, nr_of_templates_ptr, 1.0);
		reduceTemplatesFound(posTemplates, true, in);

		waitKey();
	}
}


/*
* Shows output of all positiv test images 
*/
void testQualitativ() {
	string line;
	ifstream list_pos("INRIAPerson\\Test\\pos.lst");
	//getline(list_pos, line);
	while (getline(list_pos, line)) {
		string folder = "INRIAPerson";
		string in = folder + "/" + line;
		cout << in << endl;
		int nr_of_templates = 0;
		int* nr_of_templates_ptr = &nr_of_templates;

		vector<templatePos> posTemplates;
		posTemplates = multiscaleImg(in, nr_of_templates_ptr, 1.0);
		reduceTemplatesFound(posTemplates, true, in);

		waitKey();
	}
	list_pos.close();
}

/* Task 1.5 + 3.1
*
* multiscale sliding window approach
* 
* @returns: all templates detected as human
* @para file: file of the image, which should be tested
* @nr_of_templates_ptr: int-pointer to count how much detections are in this picture
* @assumed_positiv: score which a template needs minimum to be a positiv detection
*
*/
vector<templatePos> multiscaleImg(string file, int* nr_of_templates_ptr, float assumed_positiv) {
	Mat img = imread(file);

	int factorx = TEMPLATE_WIDTH / TEMPLATE_WIDTH_CELLS;
	int factory = TEMPLATE_HEIGHT / TEMPLATE_HEIGHT_CELLS;

	int count = 0;
	vector<templatePos> posTemplates;
	CvSVM SVM;
	SVM.load(SVM_LOCATION);

	assert(!img.empty());

	double scale = pow(2.0, 1.0 / LAMBDA);
	double akt_width = img.cols;
	double akt_height = img.rows;
	int int_akt_height = floor(akt_height);
	int int_akt_width = floor(akt_width);
	double hig_scale = 1;
	Mat neuimg = img.clone();

	//scale down every loop
	while (floor(akt_width) >= TEMPLATE_WIDTH && floor(akt_height) >= TEMPLATE_HEIGHT) {
		//octave full
		if (count % LAMBDA == 0) {
			double help = pow(2, count / LAMBDA);
			akt_width = img.cols / help;
			akt_height = img.rows / help;
		}
		else {
			akt_width = akt_width / scale;
			akt_height = akt_height / scale;
		}
		int_akt_height = floor(akt_height);
		int_akt_width = floor(akt_width);

		//resize
		Mat m(int_akt_height, int_akt_width, CV_8UC3, Scalar(0, 0, 0));
		resize(img, m, m.size(), 0, 0, INTER_LINEAR);

		//compute HOG for every size
		vector<int> dims;
		double*** hog = computeHoG(m, CELL_SIZE, dims);

		vector<Point> temp_pos;
		vector<Point> real_temp_pos;
		vector<Point> real_temp_size;
		int counter = 0;
		double calc_size = (int_akt_height * int_akt_width * 2) / (TEMPLATE_WIDTH * TEMPLATE_HEIGHT);
		temp_pos.resize(calc_size);
		real_temp_pos.resize(calc_size);
		real_temp_size.resize(calc_size);

		if (dims.at(0) > TEMPLATE_HEIGHT_CELLS && dims.at(1) > TEMPLATE_HEIGHT_CELLS) {
			int template_count = 1;

			for (int i = 0; i + TEMPLATE_HEIGHT_CELLS < dims.at(0); i += floor(TEMPLATE_HEIGHT_CELLS / 4)) {
				for (int j = 0; j + TEMPLATE_WIDTH_CELLS < dims.at(1); j += floor(TEMPLATE_WIDTH_CELLS / 4)) {
					templatePos pos;
					pos.x = j*hig_scale*factorx;
					pos.y = i*hig_scale*factory;
					pos.scale = hig_scale;

					float* featureTemplate1D = compute1DTemplate(hog, dims, j, i, 0);

					Mat sampleTest(1, (TEMPLATE_WIDTH_CELLS)*(TEMPLATE_HEIGHT_CELLS)*HOG_DEPTH, CV_32FC1);
					//copy values of template to Matrix
					for (int k = 0; k < sampleTest.cols; k++) {
						sampleTest.at<float>(0, k) = featureTemplate1D[k];
					}

					float score = SVM.predict(sampleTest, true);
					pos.score = score;

					//copy templates with detections of peoples to output
					if (score > assumed_positiv) {
						posTemplates.push_back(pos);
						real_temp_pos[counter] = Point(pos.x, pos.y);
						real_temp_size[counter] = Point(TEMPLATE_WIDTH * hig_scale + pos.x, TEMPLATE_HEIGHT * hig_scale + pos.y);
						rectangle(neuimg, real_temp_pos[counter], real_temp_size[counter], CV_RGB(255, 255, 0), 1, 8);
						int baseline = 0;
						int size = getTextSize("blubb", CV_FONT_HERSHEY_SIMPLEX, TEMPLATE_WIDTH * hig_scale / 500, 1, &baseline).height;
						String selection_score = "Selection Score: " + to_string(score);
						putText(neuimg, selection_score, Point(pos.x + 2, pos.y + size + 2), CV_FONT_HERSHEY_SIMPLEX, TEMPLATE_WIDTH * hig_scale / 500, cvScalar(0, 255, 0), 1, CV_AA);
						String overlap = "Overlap: ";
						putText(neuimg, overlap, Point(pos.x + 2, pos.y + size * 2 + 4), CV_FONT_HERSHEY_SIMPLEX, TEMPLATE_WIDTH * hig_scale / 500, cvScalar(0, 255, 0), 1, CV_AA);
					}
					template_count++;
					(*nr_of_templates_ptr)++;
					free(featureTemplate1D);
				}
			}
		}
		count++;
		hig_scale *= scale;

		//destroy at end of each scale
		destroy_3Darray(hog, dims[0], dims[1]);
	}

	imshow("found_bevor_reducing", neuimg);
	return posTemplates;
}

/* Task 3.3 + 3.4
*
* Reduces the positives templates, so they don't overlap more than 20% and more than N in total
*
* @returns: maximum N "best" templates
* @param posTemplates: all positiv templates
* @param showOutput: if the output should be shown
* @param file: image we are working on
*
*/
vector<templatePos> reduceTemplatesFound(vector<templatePos> posTemplates, bool showOutput, string file) {

	Mat img = imread(file);

	if (posTemplates.empty() && showOutput) {
		showBoundingBox(img, file);
		imshow("Picture-after-reduction", img);
		String out = "QualitativOutput\\" + file + ".png";
		imwrite(out, img);
		waitKey();
	}

	if (!posTemplates.empty()) {
		vector<templatePos> nonOverlappingTemplates;
		nonOverlappingTemplates.push_back(posTemplates[0]);

		//iterate over all input templates
		for (vector<int>::size_type i = 1; i != posTemplates.size(); i++) {

			Point p1 = Point(posTemplates[i].x, posTemplates[i].y);
			Point p2 = Point(posTemplates[i].x + posTemplates[i].scale*TEMPLATE_WIDTH, posTemplates[i].y + posTemplates[i].scale*TEMPLATE_HEIGHT);

			std::vector<int> points_new = std::vector<int>(4, 0);
			points_new.at(0) = p1.x;
			points_new.at(1) = p1.y;
			points_new.at(2) = p2.x;
			points_new.at(3) = p2.y;
			bool add_new = false;

			//iteterate over all non-overlapping templates
			for (vector<int>::size_type j = 0; j != nonOverlappingTemplates.size(); j++) {
				Point p1_old = Point(nonOverlappingTemplates[j].x, nonOverlappingTemplates[j].y);
				Point p2_old = Point(nonOverlappingTemplates[j].x + nonOverlappingTemplates[j].scale*TEMPLATE_WIDTH, nonOverlappingTemplates[j].y + nonOverlappingTemplates[j].scale*TEMPLATE_HEIGHT);

				vector<int> points_old = std::vector<int>(4, 0);
				points_old.at(0) = p1_old.x;
				points_old.at(1) = p1_old.y;
				points_old.at(2) = p2_old.x;
				points_old.at(3) = p2_old.y;

				double overlap = ComputeOverlap(points_new, points_old);
				if (overlap > 0.2) {
					add_new = true;
					//if two templates overlap, add the better one (depending on score)
					if (posTemplates[i].score > nonOverlappingTemplates[j].score) {
						nonOverlappingTemplates.push_back(posTemplates[i]);
						nonOverlappingTemplates.erase(nonOverlappingTemplates.begin() + j);
					}
					else {
						continue;
					}
				}
			}

			if (add_new == false) {
				nonOverlappingTemplates.push_back(posTemplates[i]);
			}
		}

		//Reduce to maximum N templates
		if (nonOverlappingTemplates.size() > max_templates) {
			sort(nonOverlappingTemplates.begin(), nonOverlappingTemplates.end(), compareByScore);
			for (int i = 0; nonOverlappingTemplates.size() > max_templates; i++) {
				nonOverlappingTemplates.erase(nonOverlappingTemplates.begin());
			}
		}

		//Delete dublicates
		sort(nonOverlappingTemplates.begin(), nonOverlappingTemplates.end(), sortXYScale);
		nonOverlappingTemplates.erase(unique(nonOverlappingTemplates.begin(), nonOverlappingTemplates.end(), compareTemplatePos), nonOverlappingTemplates.end());

		if (showOutput) {
			visualize(nonOverlappingTemplates, file);
		}

		return nonOverlappingTemplates;
	}

}

/* Task 3.5
*
* Visualizes the positiv templates
* @param nonOverlappingTemplates: templates to visulaize
* @param file: original image
*
*/
void visualize(vector<templatePos> nonOverlappingTemplates, string file) {
	Mat img = imread(file);
	//Visualization Output 3.5
	for (vector<int>::size_type j = 0; j != nonOverlappingTemplates.size(); j++) {
		if (true) {
			vector<int> boundingBoxes = getBoundingBoxes(file);
			templatePos pos = nonOverlappingTemplates[j];
			Point p1_old = Point(pos.x, pos.y);
			Point p2_old = Point(pos.x + pos.scale*TEMPLATE_WIDTH, pos.y + pos.scale*TEMPLATE_HEIGHT);

			float overlap = getOverlap(boundingBoxes, p1_old, p2_old);
			Scalar color;
			if (overlap <= OVERLAP_CORRECT) {
				color = cvScalar(0, 0, 255);
			}
			else {
				color = cvScalar(50, 200, 50);
			}

			rectangle(img, p1_old, p2_old, color, 2, 8);
			String selection_score = "Selection Score: " + to_string(pos.score);
			int baseline = 0;
			int size = getTextSize("blubb", CV_FONT_HERSHEY_SIMPLEX, TEMPLATE_WIDTH * pos.scale / 200, 1, &baseline).height;
			putText(img, selection_score, Point(pos.x + 2, pos.y + size + 2), CV_FONT_HERSHEY_SIMPLEX, TEMPLATE_WIDTH * pos.scale / 300, color, 1, CV_AA);

			String overlap_out = "Overlap: " + to_string(overlap);
			putText(img, overlap_out, Point(pos.x + 2, pos.y + size * 2 + 4), CV_FONT_HERSHEY_SIMPLEX, TEMPLATE_WIDTH * pos.scale / 300, color, 1, CV_AA);
		}
	}

	if (true) {
		imshow("Picture-after-reduction", img);
	}
}

/*
* Computes the maximal overlap with any of the truth bounding boxes in a picture for a given box
*
* @returns: maximal overlap
* @param truth: all truth bounding boxes of a picture
* @param p1: (xMin, yMin) of the given box
* @param p2: (xMax, yMax) of the given box
*
*/
float getOverlap(vector<int> truth, Point p1, Point p2) {
	std::vector<int> detected = std::vector<int>(4, 0);
	detected.at(0) = p1.x;
	detected.at(1) = p1.y;
	detected.at(2) = p2.x;
	detected.at(3) = p2.y;
	int i = 0;
	float overlap = 0;
	float overlap_temp = 0;
	while (truth.size() - i > 3) {
		std::vector<int> truth_i = std::vector<int>(4, 0);
		truth_i.at(0) = truth.at(0 + i);
		truth_i.at(1) = truth.at(1 + i);
		truth_i.at(2) = truth.at(2 + i);
		truth_i.at(3) = truth.at(3 + i);
		overlap_temp = ComputeOverlap(truth_i, detected);
		if (overlap_temp >= overlap) {
			overlap = overlap_temp;
		}
		i += 4;
	}
	return overlap;
}

/*
* Calculates if a bounding box was found of any of the detected templates in a picture
*
* @returns: the higest overlap of the bounding box with all of the found templates
* @param allTempates: all templates which are detected as a human in a given picture
* @param truth: all truth bounding boxes of the picture
* @param which_bounding_box: which of the truth bounding boxes we are testing
* @param min_score a template must have to be considered as positiv
*
*/
float isFound(vector<templatePos> allTemplates, vector<int> truth, int which_bounding_box, float min_score) {

	float overlap = 0;
	float overlap_temp = 0;
	vector<int> truth_bb = vector<int>(4, 0);
	truth_bb.at(0) = truth.at(4 * which_bounding_box);
	truth_bb.at(1) = truth.at(4 * which_bounding_box + 1);
	truth_bb.at(2) = truth.at(4 * which_bounding_box + 2);
	truth_bb.at(3) = truth.at(4 * which_bounding_box + 3);
	for (vector<templatePos>::const_iterator j = allTemplates.begin(); j != allTemplates.end(); ++j) {
		if ((*j).score > min_score) {
			templatePos pos = (*j);
			Point p1 = Point(pos.x, pos.y);
			Point p2 = Point(pos.x + pos.scale*TEMPLATE_WIDTH, pos.y + pos.scale*TEMPLATE_HEIGHT);

			std::vector<int> allTemplates_i = std::vector<int>(4, 0);
			allTemplates_i.at(0) = p1.x;
			allTemplates_i.at(1) = p1.y;
			allTemplates_i.at(2) = p2.x;
			allTemplates_i.at(3) = p2.y;
			overlap_temp = ComputeOverlap(truth_bb, allTemplates_i);
			if (overlap_temp >= overlap) {
				overlap = overlap_temp;
			}
		}
	}
	return overlap;
}

/*
* Tests if two templatePos describe the same template
*
* @returns: true if they are the same
* @param pos1: first templatePos for comparison
* @param po2: second templatePos for comparison
*
*/
bool compareTemplatePos(templatePos pos1, templatePos pos2) {
	if (pos1.x == pos2.x && pos1.y && pos2.y && pos1.scale == pos2.scale) {
		return true;
	}
	else return false;
}

/*
* Compares templates by position (Xmin, Ymin)
*
* returns: true if pos1 is smaller
* @param pos1: first templatePos for comparison
* @param po2: second templatePos for comparison
*
*/
bool sortXYScale(templatePos pos1, templatePos pos2) {
	if (pos1.x <= pos2.x) {
		return true;
	}
	else if (pos1.x == pos2.x) {
		if (pos1.y <= pos2.y) {
			return true;
		}
		else if (pos1.y == pos2.y) {
			if (pos1.scale <= pos2.scale) {
				return true;
			}
		}
	}
	return false;
}

