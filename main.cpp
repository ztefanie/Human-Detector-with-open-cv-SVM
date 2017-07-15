#include "main.h"
#include "utils.h"
#include "tests.h"
#include "optimizeSVM.h"
#include "testSVM.h"
#include "DET.h"
#include "prepareDET.h"



// Function to print the output to a log instead of printing to the commandline
void logOutput() {
	string out_file = "log_" + getTimeLog() + ".txt";
	freopen(out_file.c_str(), "w", stdout);
}


/*
*	Main Function containing all tasks for the Project
*
*	Uncomment what you what to test
*/
int main(int argc, char* argv[])
{
	//find_hardPositives();
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
	*	first line for iterating over all positiv files from beginning, second line for random picture order
	*
	*/
	//testQualitativ();
	testQualitativRand();

	/* Task 3.6
	*
	*	Quantitativ Evaluation
	*
	*	uses Functions from prepareDET-class to write miss-rates and fppw for different detection thresholds
	*	after executing this function, run the python-script to generate the plot.
	*
	*/
	//createDETfile();


	cout << endl << "finished" << endl;
	getchar();
}

