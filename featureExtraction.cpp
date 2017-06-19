#include "featureExtraction.h"
#include "main.h"
#include "hog.h"


using namespace std;
using namespace cv;

//Task 1.4
double*** compute3DTemplate(double*** hog, const std::vector<int> &dims, int grid_pos_x, int grid_pos_y) {
	//Test if input is valid
	assert(grid_pos_x >= 0);
	assert(grid_pos_y >= 0);
	assert(grid_pos_y + TEMPLATE_HEIGHT_CELLS <= dims.at(0));
	assert(grid_pos_x + TEMPLATE_WIDTH_CELLS <= dims.at(1));

	//init array
	double*** featureRepresentation = 0;

	//alloc space for 3D-array
	featureRepresentation = new double**[TEMPLATE_HEIGHT_CELLS];
	for (int i = 0; i < TEMPLATE_HEIGHT_CELLS; i++) {
		featureRepresentation[i] = new double*[TEMPLATE_WIDTH_CELLS];
		for (int j = 0; j < TEMPLATE_WIDTH_CELLS; j++) {
			featureRepresentation[i][j] = new double[HOG_DEPTH];
		}
	}

	for (int y = 0; y < TEMPLATE_HEIGHT_CELLS; y++) {
		for (int x = 0; x < TEMPLATE_WIDTH_CELLS; x++) {
			for (int b = 0; b < HOG_DEPTH; b++) {
				//copy template-part of hog to output array
				double value = hog[y + grid_pos_y][x + grid_pos_x][b];
				featureRepresentation[y][x][b] = value;
			}
		}
	}
	return featureRepresentation;
}

double* compute1DTemplate(double*** hog, const std::vector<int> &dims, int grid_pos_x, int grid_pos_y, int scale) {
	//Test if input is valid
	assert(grid_pos_x >= 0);
	assert(grid_pos_y >= 0);
	assert(grid_pos_y + TEMPLATE_HEIGHT_CELLS <= dims.at(0));
	assert(grid_pos_x + TEMPLATE_WIDTH_CELLS <= dims.at(1));

	//allocate 1D array
	double* featureRepresentation = (double*)malloc(sizeof(double)*TEMPLATE_HEIGHT_CELLS*TEMPLATE_WIDTH_CELLS*HOG_DEPTH);
	for (int y = 0; y < TEMPLATE_HEIGHT_CELLS; y++) {
		for (int x = 0; x < TEMPLATE_WIDTH_CELLS; x++) {
			for (int b = 0; b < HOG_DEPTH; b++) {
				//copy values from 3D to 1D array
				double value = hog[y + grid_pos_y][x + grid_pos_x][b];
				featureRepresentation[y*TEMPLATE_WIDTH_CELLS*HOG_DEPTH + x*HOG_DEPTH + b] = value;
			}
		}
	}
	return featureRepresentation;
}


//Task 1.5 - mostly the code from fabio (see a1.5.cpp)
void multiscale(Mat img) {

	int count = 0;

	assert(!img.empty());

	double scale = pow(2.0, 1.0 / LAMBDA);

	double akt_width = img.cols;
	double akt_height = img.rows;
	int int_akt_height = floor(akt_height);
	int int_akt_width = floor(akt_width);
	double hig_scale = 1;
	
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

		//round
		int_akt_height = floor(akt_height);
		int_akt_width = floor(akt_width);

		//resize
		Mat m(int_akt_height, int_akt_width, CV_8UC3, Scalar(0, 0, 0));
		resize(img, m, m.size(), 0, 0, INTER_CUBIC);
		cout << int_akt_height << " " << int_akt_width << endl;

		//compute HOG for every size
		vector<int> dims;
		double*** hog = computeHoG(m, CELL_SIZE, dims);
		//double*** hog = extractHOGFeatures("INRIAPerson\\Train\\pos", "crop_000603.png", dims);
		Mat out = visualizeGradOrientations(hog, dims);
		String pic = "Gradients at scale: " + to_string(count);
		imshow(pic, out);
		/*dims[0] = int_akt_width;
		dims[1] = int_akt_height;
		vector<int> dims_template;*/
		//double*** featureTemplate = compute3DTemplate(hog, dims, 0, 0);

		//erstmal 50% überlappung; bei 75% durch 4 teilen:
		int counter = 0;

		for (int i = 0; i + TEMPLATE_HEIGHT <= int_akt_height; i += floor(TEMPLATE_HEIGHT / 2)) {
			for (int j = 0; j + TEMPLATE_WIDTH <= int_akt_width; j += floor(TEMPLATE_WIDTH / 2)) {
				//compute templates at every position
				//double*** featureTemplate = compute3DTemplate(hog, dims, 0, 0);

					//Show (just for testing)
				/*if (count == 5) {
					vector<int> dims2 = vector<int>(3);
					dims2[0] = TEMPLATE_HEIGHT_CELLS;
					dims2[1] = TEMPLATE_WIDTH_CELLS;
					dims2[2] = HOG_DEPTH;
					Mat out2 = visualizeGradOrientations(featureTemplate, dims2);
					imshow("Iterate over scale = 5", out2);
				}*/
			}
		}
		count++;
		hig_scale *= scale;

		//destroy at end of each scale
		destroy_3Darray(hog, dims[0], dims[1]);
	}
}

