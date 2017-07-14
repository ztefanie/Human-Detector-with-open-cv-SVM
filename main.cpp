#include "main.h"
#include "tests.h"
#include "optimizeSVM.h"
#include "testSVM.h"
#include "DET.h"
#include "prepareDET.h"
#include <ctime>
#include <fstream>

// Function to print the output to a log instead of printing to the commandline
void logOutput() {
	time_t t = time(0);   // get time now
	struct tm * now = localtime(&t);
	stringstream ss;
	ss << "log_" << (now->tm_year + 1900) << '-'
		<< (now->tm_mon + 1) << '-'
		<< now->tm_mday << "-"
		<< now->tm_hour << "-" << now->tm_min
		<< ".txt";
string out_file = ss.str();

freopen(out_file.c_str(), "w", stdout);
}


/*
*	Main Function containing all tasks for the Project
*
*	Uncomment what you what to test
*/
int main(int argc, char* argv[])
{
	//Uncomment next line if you wish output to log file
	//logOutput();

	//Task 1.1
	//testDrawBoundingBox();

	//Task 1.2
	//testOverlapBoundingBox();

	/*	Task 1.3 + Task 1.4
	*
	*	see also
	*	getTemplate-Method from trainSVM.cpp, which extracts a random template of negativ images
	*	get1DTemplateFromPos-Method from featureExtraction.cpp, which extracts templates with humans from positiv images
	*
	*	extractHOGFeatures from utils.cpp extracts complete HoG of a image
	*	compute3DTempalate and compute1DTemplate from featureExtraction.cpp create a template of a image at a specific position
	*
	*	uncomment next to lines to see the visualizable part of the task
	*/

	//testHog(); //shows a HoG of a picture	
	//test3DTemplate(); //shows the HoG of a template	
	//int last; Mat in;	 //shows how positive templates are extracted
	//get1DTemplateFromPos("\\Train\\pos\\crop_000607.png", in, &last, true); 



	/* Task 1.5
	*
	*	the method used for detecting windows is multiscaleImg() in testSVM.cpp
	*	testDownScale() uses the same scaling process but visualizes the output -> uncomment next line to see it and press any key to go to next scale
	*
	*/
	//testDownScale();


	/* Task 2.1
	*
	*	Function that trains SVM.
	*	Set input paramter to false = first training  or true = retraining
	*
	*/
	//SVMtrain(false);

	/* Task 2.2
	*
	*	Function that retrains SVM.
	*	Uses the find_hardNegatives()-Method from optimzeSVM
	*
	*/
	//SVMtrain(true);

	/* Task 3.1 - 3.4
	*
	*	Qualitativ Evaluation
	*
	*	uses the multiscaleImg Function to find all templates with a score big enough
	*	and reduceTemplatesFound Function to realize non-maxima suppression and remove overlapping templates
	*
	*	Uncomment line and use any key to get the results for the next test image
	*
	*/
	//testQualitativ();
	cout << "Logging?     y/n" << endl;

	if (getchar() == 'y') {
		cout << "will be loggt..." << endl << endl;
		logOutput();
	}
	else {
		cout << "no Logging" << endl << endl;
	}

	cout << "Human Detection with Histograms of Oriented Gradients" << endl;
	int in = 0;
	while (in == 0) {
		while (getchar() != '\n');
		cout << endl << "choose:" << endl;
		cout << "(1) train SVM" << endl;
		cout << "(2) retrain SVM" << endl;
		cout << "(3) get DET file" << endl;
		cout << "(4) go though positiv test Fieles" << endl;
		cout << "(5) go though positiv training Fieles" << endl;
		cout << "(6) go though negative test Fieles" << endl;
		cout << "//(7) go though negative training Fieles" << endl;
		cout << "(8) Presentation" << endl;
		cout << "(9) End" << endl;
		ifstream f1(SVM_LOCATION);
		ifstream f2(SVM_2_LOCATION);

		switch (getchar()) {
		case '1':
			SVMtrain(false);
			break;
		case '2':
			SVMtrain(true);
			break;
		case '3':
			if (f1.good() && f2.good()) {
				createDETfile();
			}
			else {
				cout << "no SVM found" << endl;
			}
			break;
		case '4':
			if (f2.good()) {
				testQualitativ();
			}
			else {
				cout << "no SVM found" << endl;
			}
			break;
		case '5':
			if (f2.good()) {
				testTraining();
			}
			else {
				cout << "no SVM found" << endl;
			}
			break;
		case '6':
			if (f2.good()) {
				negaivTest();
			}
			else {
				cout << "no SVM found" << endl;
			}
			break;
		case '7':
			if (f2.good()) {
				negaivTrainTest();
			}
			else {
				cout << "no SVM found" << endl;
			}
			break;
		case '8':
			if (f2.good()) {
				presentation();
			}
			else {
				cout << "no SVM found" << endl;
			}
			break;
		case '9':
			cout << "bye" << endl;
			in = 1;
			break;
		case '0':
			SVMtrain(false);
			SVMtrain(true);
			testQualitativ();
			createDETfile();
			break;
		default:
			cout << "wrong input" << endl;
			break;

		}
		f1.close();
		f2.close();
	}
	//Task 3.6
	//createDETfile();

	cout << endl << "finished" << endl;
	
	//waitKey();
	while (getchar() != '\n');
	getchar();

}

