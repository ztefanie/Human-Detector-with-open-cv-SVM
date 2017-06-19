#include "main.h"

int main(int argc, char* argv[]) {

	//Task 1.1
	//testDrawBoundingBox();

	//Task 1.2
	//testOverlapBoundingBox();

	//Task 1.3
	//testHog();
	
	//Task 1.4
	//test3DTemplate(); 
					  //1DTemplate not tested

	//Task 1.5
	//testDownScale();
	testMultiscale();	//computes HoG for every size (working)
						//and creates templates on relevant positions - overlap 0.5 (in progress)
	
	waitKey();

	return 0;
}