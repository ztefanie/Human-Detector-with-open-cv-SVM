#include <Functions.h>

int main()
{
	//Variables
	string inputstring, inputstring1;
	int input, kernel, height, width, spare;
	width = height = 512;
	int n = 100;
	
	while (true)
	{
		colorConsole(15);
		cout << "Bitte waehlen Sie eine Aufgabe von Blatt 4 aus:" << endl;
		cout << "(1)" << endl;
		cout << "(2)" << endl;
		getline(cin, inputstring1);
		input = atoi(inputstring1.c_str());

		if (input == 1)
		{
			spare = 25;
			break;
		}

		if (input == 2)
		{
			colorConsole(12);
			cout << "Fehler, Aufgabe 2 ist noch nicht implementiert. Es wird automatisch Aufgabe 1 verwendet!" << endl;
			spare = 25;
			break;
		}

		colorConsole(12);
		cout << "Fehler, Ihre Eingabe war nicht korrekt!" << endl;
	}

	while (true)
	{
		colorConsole(15);
		cout << "Bitte waehlen Sie einen SVM Kernel Typ aus:" << endl;
		cout << "(1) Linear" << endl;
		cout << "(2) RBF" << endl;
		getline(cin, inputstring);
		kernel = atoi(inputstring.c_str());

		if (kernel == 1)
		{
			break;
		}
		
		if (kernel == 2)
		{
			colorConsole(12);
			cout << "Fehler, RBF ist noch nicht implementiert. Es wird automatisch Linear verwendet!" << endl;
			kernel = 1;
			colorConsole(15);
			break;
		}

		colorConsole(12);
		cout << "Fehler, Ihre Eingabe war nicht korrekt!" << endl;
	}

	//1 a) (also used in trainSVM from 1 b))
	/*Mat trainingDataMat(n, 2, CV_32FC1);
	int* labels = new int[n]();
	Mat img = drawSet(createSimpleTrainingSet(n, width, height, spare, trainingDataMat, labels));

	string Name = "SimpleTrainingSet.jpg";
	imshow(Name, img);
	imwrite(Name, img);*/

	//1 b)
	Mat img = trainSVM(n, width, height, spare, kernel);
	string Name = "SimpleTrainingSet.jpg";
	imshow(Name, img);
	imwrite(Name, img);
	
	//1 c)
	Mat svmClass = testSVM(width, height) + img;
	
	string Name1 = "SVM.jpg";
	imshow(Name1, svmClass);
	imwrite(Name1, svmClass);

	waitKey();
	destroyAllWindows();
	return 0;
}
