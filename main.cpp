#include "main.h"

int main(int argc, char* argv[])
{
	firstStepTrain();
	
	//test3DTemplate();

	cout << endl << "finished" << endl;
	waitKey();

	/*string inputstring;

	while (true)
	{
		colorConsole(15);
		cout << endl;
		cout << "Bitte waehlen Sie eine Aufgabe aus: " << endl;
		cout << "1 - Task 1.1" << endl;
		cout << "2 - Task 1.2" << endl;
		cout << "3 - Task 1.3" << endl;
		cout << "4 - Task 1.4" << endl;
		cout << "5 - Task 1.5" << endl;
		cout << "6 - Task 2.1" << endl;
		cout << "q - Programm beenden" << endl;
		//cout << "--------------------------------------------" << endl;
		//cout << "(Alternativ koennen Sie durch Eingabe von ";
		//colorConsole(10);
		//cout << "d";
		//colorConsole(15);
		//cout << " oder durch einfaches Druecken der Enter-Taste auch eine Standardaufruf starten)" << endl;
		getline(cin, inputstring);

		//if (convertStringToLower(inputstring) == "default" || inputstring == "")
		//{
		//	inputstring = "eye.png";
		//}

		switch (inputstring[0])
		{
		case '1':
			system("cls");
			cout << "Task 1.1 ausgewaehlt" << endl;
			testDrawBoundingBox();
			break;
		case '2':
			system("cls");
			cout << "Task 1.2 ausgewaehlt" << endl;
			testOverlapBoundingBox();
			break;
		case '3':
			system("cls");
			cout << "Task 1.3 ausgewaehlt" << endl;
			testHog();
			break;
		case '4':
			system("cls");
			cout << "Task 1.4 ausgewaehlt" << endl;
			test3DTemplate();
			//1DTemplate not tested
			break;
		case '5':
			system("cls");
			cout << "Task 1.5 ausgewaehlt" << endl;
			testDownScale();
			testMultiscale(); //computes HoG for every size (working) and creates templates on relevant positions - (overlap 0.5)
			break;
		case '6':
			system("cls");
			cout << "Task 2.1 ausgewaehlt" << endl;
			firstStepTrain();
			break;
			//case 'd':
				//break;
		case 'q':
			return 0;
		default:
			system("cls");
			colorConsole(12);
			cout << "Fehler, Sie haben eine falsche Auswahl getroffen!" << endl;
		}
	}*/
}
