#ifndef MAIN_H_INCLUDED
#define MAIN_H_INCLUDED

#define _USE_MATH_DEFINES

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>

#include "hog.h"
#include "utils.h"
#include "featureExtraction.h"
#include "trainSVM.h"
#include "tests.h"

using namespace std;
using namespace cv;



#define CELL_SIZE				8		// size of a cell (in pixels) - best option according to Dalal and Triggs Paper
#define LAMBDA					5		// steps in one octave
#define TEMPLATE_HEIGHT			128		// the height of a template (in pixels)
#define TEMPLATE_WIDTH			64		// the width of a template (in pixels)

// the width of a window (in pixels)
#define TEMPLATE_HEIGHT_CELLS	16		// the height of a template (in cells)
#define TEMPLATE_WIDTH_CELLS	8		// the width of a window (in cells)
#define HOG_DEPTH				32		// dims_z from the hog-implementation

#define SVM_LOCATION			"SVM.xml"
#define SVM_2_LOCATION			"SVM_2.xml"
#define LIST_POS				"INRIAPerson\\Train\\pos.lst"
#define LIST_POS_NORM			"INRIAPerson\\train_64x128_H96\\pos.lst"
#define LIST_NEG				"INRIAPerson\\Train\\neg.lst"
#define MAX_HARD_NEG			1198	// Number-of-positives minus Number-of-negatives		

#define max_templates			15		// maximum number of templates that can be found in one picture
#define ASSUMED_POSITIV			1		// min score a template needs for positiv detection
#define OVERLAP_CORRECT			0.4

#endif


