#include "Functions.h"

//define general constants
//#define PI 3.141592653589793
//#define E 2.7182818284

//define function specific constants

//--------------------------------

//Check if running on Windows - if not ->no colors in console window!
#ifdef _WIN32
HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

void colorConsole(int color)
{
	SetConsoleTextAttribute(hConsole, color);
}
#endif

//Functions
//Sonstiges
//void colorConsole(int color)

string convertStringToLower(string str)
{
	int i = 0;
	const char* cstr = str.c_str();
	string c;

	while (cstr[i])
	{
		c += tolower(cstr[i]);
		i++;
	}

	const char* retstr = c.c_str();
	return retstr;
}

int newRandomintGenerator(int min, int max)
{
	random_device rd;
	mt19937 mt(rd());
	uniform_real_distribution<> dist(min, max);
	return dist(mt);
}

double newRandomdoubleGenerator(double min, double max)
{
	random_device rd;
	mt19937 mt(rd());
	uniform_real_distribution<double> dist(min, max);
	return dist(mt);
}

//--------------------------------

//Blatt00 - 0.3
int RandomintGenerator(int MAX, int MIN)
{
	return MIN + (rand() % int(MAX - MIN + 1));
}

double RandomdoubleGenerator(double MAX, double MIN)
{
	double f = (double)rand() / RAND_MAX;
	return MIN + f * (MAX - MIN);
}

int WriteText(string path, int n, string text)
{
	//Variables
	double width, height, fontsize, xmax, ymin, ymax;

	//Open Image
	Mat img = imread(path);

	//Calculate size
	width = img.size().width;
	height = img.size().height;
	ymax = height - 4;

	for (int i = 1; i <= n; i++)
	{
		fontsize = RandomdoubleGenerator(3.0, 0.5);

		if (fontsize <= 3.0)
		{
			xmax = 0.1 * width;
			ymin = 50;
		}
		else if (fontsize <= 2.0)
		{
			xmax = 0.4 * width;
			ymin = 30;
		}
		else
		{
			xmax = 0;
			ymin = 50;
		}

		double posX = RandomdoubleGenerator(xmax, 0);
		double posY = RandomdoubleGenerator(ymax, ymin);
		int colorR = RandomintGenerator(255, 1);
		int colorG = RandomintGenerator(255, 1);
		int colorB = RandomintGenerator(255, 1);

		putText(img, text, cvPoint(posX, posY), FONT_HERSHEY_COMPLEX_SMALL, fontsize, cvScalar(colorR, colorG, colorB), 1);
	}

	//Generate savename of Image
	std::ostringstream oss;
	oss << "Lenna-" << n << ".png";
	std::string Name = oss.str();

	//Show Image
	imshow(Name, img);

	//Save Image
	imwrite(Name, img);

	waitKey();
	destroyAllWindows();

	return 0;
}

//--------------------------------

//Blatt00 - 0.4
//Check if file exists
bool file_exists(string fn)
{
	return experimental::filesystem::exists(fn);
}

//Create custom string with string + variable
string compare(string Text, string Variable)
{
	ostringstream oss;
	oss << Text << Variable;
	return oss.str();
}

//Parse double to string
string doubleTostring(double val)
{
	ostringstream strs;
	strs << val;
	return strs.str();
}

//Parse int to string
string intTostring(int val)
{
	ostringstream strs;
	strs << val;
	return strs.str();
}

//Calculate varianz parts
double caclvarianz(double i, double n)
{
	return pow((i - n), 2);
}

//Calculate meidan in vector
double CalcMHWScore(vector<int> scores)
{
	double median;
	size_t size = scores.size();

	sort(scores.begin(), scores.end());

	if (size % 2 == 0)
	{
		median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
	}
	else
	{
		median = scores[size / 2];
	}

	return median;
}

//--------------------------------

//Blatt01 - 1.1
void createHistogram(Mat img, int b, string name)
{
	int numberOfChannels = img.channels();
	int hist_w = 512;
	int hist_h = 400;
	double bin_w = cvRound((double)hist_w / b);
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));
	std::vector<double> histogram(b, 0);

	//Read value of every pixel in image
	for (int y = 0; y < img.rows; y++)
	{
		uchar* imagePtr = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			for (int c = 0; c < numberOfChannels; c++)
			{
				uchar colorVal = imagePtr[x * numberOfChannels + c];
				histogram[b * int(colorVal) / 256]++;
			}
		}
	}

	//Find maximum
	double max = histogram[0];
	for (int i = 1; i < b; i++)
	{
		if (max < histogram[i])
		{
			max = histogram[i];
		}
	}

	//Normalize histogram
	for (int i = 0; i < b; i++)
	{
		histogram[i] = (double(histogram[i]) / max) * img.rows;
	}

	//Draw lines for histogram
	for (int i = 0; i < b; i++)
	{
		line(histImage, Point2d(bin_w * i, hist_h), Point2d(bin_w * i, hist_h - histogram[i]), Scalar(0, 0, 0), 1, 8, 0);
	}

	//Display original image
	imshow(name, img);

	string str = "Histogram-" + to_string(b) + "-" + name;
	//Display and save histogram
	namedWindow(str, CV_WINDOW_AUTOSIZE);
	imshow(str, histImage);
	imwrite(str, histImage);

	waitKey();
	destroyAllWindows();
}

void enhanceBrightnessAndContrast(Mat image, int b, string name)
{
	string con;
	string bri;

	//Define values
	cout << "Bitte einen Wert fuer den Kontrast eingeben [1.0-3.0]: ";
	getline(cin, con);

	cout << "Bitte einen Wert fuer die Helligkeit eingeben [0-255]: ";
	getline(cin, bri);

	//Convert values or set default values
	double contrast;
	if (!con.empty() && stod(con) >= 1.0 && stod(con) <= 3.0)
	{
		contrast = stod(con);
	}
	else
	{
		contrast = 1.0;
		cout << "Sie haben einen ungueltigen Wert als Kontrast eingegeben, es wird 1.0 - keine Veraenderung gesetzt!" << endl;
	}

	int brightness;
	if (!bri.empty() && stod(bri) >= 0 && stod(bri) <= 255)
	{
		brightness = stoi(bri);
	}
	else
	{
		brightness = 0;
		cout << "Sie haben einen ungueltigen Wert als Helligkeit eingegeben, es wird 0 - keine Veraenderung gesetzt!" << endl;
	}

	//Begin enhancement
	int numberOfChannels = image.channels();
	int hist_w = 512;
	int hist_h = 400;
	double bin_w = cvRound((double)hist_w / b);
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));
	Mat new_image = Mat::zeros(image.size(), image.type());
	std::vector<double> histogram(b, 0);

	//Read value of every pixel in image
	for (int y = 0; y < image.rows; y++)
	{
		uchar* imagePtr = image.ptr<uchar>(y);
		uchar* imagePtrnew = new_image.ptr<uchar>(y);
		for (int x = 0; x < image.cols; x++)
		{
			for (int c = 0; c < numberOfChannels; c++)
			{
				uchar value = imagePtr[x * numberOfChannels + c];
				int colorVal = contrast * int(value) + brightness;

				//Detect and correct overflow
				if (colorVal <= 255)
				{
					imagePtrnew[x * numberOfChannels + c] = colorVal;
					histogram[b * colorVal / 256]++;
				}
				else
				{
					imagePtrnew[x * numberOfChannels + c] = 255;
					histogram[b * 255 / 256]++;
				}
			}
		}
	}

	//Find maximum
	double max = histogram[0];
	for (int i = 1; i < b; i++)
	{
		if (max < histogram[i])
		{
			max = histogram[i];
		}
	}

	//Normalize histogram
	for (int i = 0; i < b; i++)
	{
		histogram[i] = (double(histogram[i]) / max) * image.rows;
	}

	//Draw lines for histogram
	for (int i = 0; i < b; i++)
	{
		line(histImage, Point2d(bin_w * i, hist_h), Point2d(bin_w * i, hist_h - histogram[i]), Scalar(0, 0, 0), 1, 8, 0);
	}

	string str1 = "Contrast-" + name;
	string str2 = "Histogram-Contrast-" + to_string(b) + "-" + name;

	//Display original image
	imshow(name, image);

	//Display and save new image
	namedWindow(str1, CV_WINDOW_AUTOSIZE);
	imshow(str1, new_image);
	imwrite(str1, new_image);

	//Display and save histogram
	namedWindow(str2, CV_WINDOW_AUTOSIZE);
	imshow(str2, histImage);
	imwrite(str2, histImage);

	waitKey();
	destroyAllWindows();
}

//--------------------------------

//Blatt02 - 2.1
Mat createPattern(int size)
{
	int width = size;
	int height = size;
	Mat img(width, height, CV_8UC1);

	double A = 127.5;
	double o = 127.5;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img.at<uchar>(y, x) = (A * (sin(0.5 * M_PI * ((pow(getValue(width, x), 2) + pow(getValue(height, y), 2)) / height))) + o);
		}
	}
	imwrite("Pattern.jpg", img);
	return img;
}

double getValue(int a, int b)
{
	return (b - ((a - 1) / 2));
}

Mat filterImg(Mat img, int kernel, int filter)
{
	if ((kernel % 2) == 0)
	{
		colorConsole(12);
		cout << "Wrong kernel size. " + to_string(kernel) + " % 2 needs to be != 0!" << endl;
		return img;
	}

	if (filter != 1 && filter != 2 && filter != 3)
	{
		string sfilter;
		while (true)
		{
			colorConsole(12);
			cout << "Please input a valid filter value! [1 = box filter, 2 = Gaussian filter, 3 = median filter]" << endl;
			getline(cin, sfilter);
			filter = atoi(sfilter.c_str());
			if (filter == 1 || filter == 2 || filter == 3)
			{
				colorConsole(15);
				break;
			}
		}
	}

	Mat filteredImg;
	img.copyTo(filteredImg);
	int offset = kernel / 2;

	for (int y = offset; y < img.rows - offset; y++)
	{
		for (int x = offset; x < img.cols - offset; x++)
		{
			switch (filter)
			{
			case 1: filteredImg.at<uchar>(y, x) = boxFilter(img, kernel, x, y);
				break;
			case 2: filteredImg.at<uchar>(y, x) = medianFilter(img, kernel, x, y);
				break;
			case 3: filteredImg.at<uchar>(y, x) = gaussianFilter(img, kernel, x, y);
				break;
			default:
				break;
			}
		}
	}

	return filteredImg;
}

uchar boxFilter(Mat img, int kernel, int x, int y)
{
	float sum = 0;

	for (int j = y - (kernel / 2); j <= y + (kernel / 2); j++)
	{
		for (int i = x - (kernel / 2); i <= x + (kernel / 2); i++)
		{
			sum += img.at<uchar>(j, i);
		}
	}

	return uchar(float(sum) / pow(kernel, 2));
}

uchar medianFilter(Mat img, int kernel, int x, int y)
{
	int length = pow(kernel, 2);
	kernel = (kernel / 2);
	int k = 0;
	vector<int> values(length);

	for (int j = y - kernel; j <= y + kernel; j++)
	{
		for (int i = x - kernel; i <= x + kernel; i++)
		{
			values[k++] = img.at<uchar>(j, i);
		}
	}

	sort(values.begin(), values.end());

	if ((length % 2) == 0)
	{
		return uchar(floor(((values.at(length / 2) + values.at((length / 2) - 1)) / 2) + 0.5));
	}

	return uchar(values.at((floor(length / 2))));
}

uchar gaussianFilter(Mat img, int kernel, int x, int y)
{
	float sum = 0;
	float sig = 1;
	kernel = (kernel / 2);

	for (int j = y - kernel; j <= y + kernel; j++)
	{
		for (int i = x - kernel; i <= x + kernel; i++)
		{
			sum += img.at<uchar>(j, i) * (1.0 / (2.0 * M_PI * pow(sig, 2))) * pow(M_E, (-1.0 * (pow(i - x, 2) + pow(j - y, 2))) / (2 * pow(sig, 2)));
		}
	}

	return uchar(sum);
}

Mat combineFourImg(Mat img1, Mat img2, Mat img3, Mat img4)
{
	if ((img1.rows == img2.rows && img1.rows == img3.rows && img1.rows == img4.rows) && (img1.cols == img2.cols && img1.cols == img3.cols && img1.cols == img4.cols))
	{
		Mat combinedImg;
		img1.copyTo(combinedImg);

		for (int y = 0; y < combinedImg.rows; y++)
		{
			for (int x = 0; x < combinedImg.cols; x++)
			{
				int rowBorder = combinedImg.rows / 2;
				int colBorder = combinedImg.cols / 2;

				if (y <= rowBorder && x > colBorder)
				{
					combinedImg.at<uchar>(y, x) = img2.at<uchar>(y, x);
				}
				else if (y > rowBorder && x <= colBorder)
				{
					combinedImg.at<uchar>(y, x) = img3.at<uchar>(y, x);
				}
				else if (y > rowBorder && x > colBorder)
				{
					combinedImg.at<uchar>(y, x) = img4.at<uchar>(y, x);
				}
			}
		}
		return combinedImg;
	}
	colorConsole(12);
	cout << "Combining isn't possible, because the image sizes aren't equal!" << endl;
	colorConsole(15);
	return img1;
}

//--------------------------------

//Blatt02 - 2.2
Mat filterImg(Mat img, int mode)
{
	Mat newimg;
	int offset = 1;

	if (mode == 4)
	{
		newimg = Mat(img.rows, img.cols, CV_32FC1);
	}
	else
	{
		newimg = Mat(img.rows, img.cols, CV_8UC1);
	}

	for (int y = offset; y < img.rows - offset; y++)
	{
		for (int x = offset; x < img.cols - offset; x++)
		{
			float val = 0;

			switch (mode)
			{
			case 1: val = int(abs(Sobel(img, x, y, 1)));
				break;
			case 2: val = int(abs(Sobel(img, x, y, 2)));
				break;
			case 3: val = int(abs(Sobel(img, x, y, 1)) + abs(Sobel(img, x, y, 2)));
				break;
			case 4: val = int(atan2(Sobel(img, x, y, 2), Sobel(img, x, y, 1)));
				break;
			default:
				break;
			}

			if (mode == 4)
			{
				newimg.at<float>(y, x) = val;
			}
			else
			{
				if (val > 255) val = 255;
				if (val < 0) val = 0;
				newimg.at<uchar>(y, x) = val;
			}
		}
	}
	return newimg;
}

int Sobel(Mat img, int x, int y, int dir)
{
	int sum = 0;
	vector<int> filter;

	if (dir == 1)
	{
		filter = {1, 0, -1, 2, 0, -2, 1, 0, -1};
	}
	else if (dir == 2)
	{
		filter = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	}

	for (int j = y - 1; j <= y + 1; j++)
	{
		for (int i = x - 1; i <= x + 1; i++)
		{
			sum += img.at<uchar>(j, i) * filter.back();
			filter.pop_back();
		}
	}
	return sum;
}

Mat Viszualize(Mat img, Mat magnitude, Mat directions)
{
	Mat bgrImg;
	cvtColor(img, bgrImg, COLOR_GRAY2BGR);

	int l = 25;
	int t = 38;
	int offset = 5;

	for (int y = offset; y < img.rows - offset; y += offset)
	{
		for (int x = offset; x < img.cols - offset; x += offset)
		{
			if (magnitude.at<uchar>(y, x) > t)
			{
				float length = l * (float(magnitude.at<uchar>(y, x)) / 255);
				int xE = length * cos(directions.at<float>(y, x));
				int yE = length * sin(directions.at<float>(y, x));
				circle(bgrImg, Point(x, y), 1, Scalar(0, 0, 255), CV_FILLED);
				line(bgrImg, Point(x - xE, y - yE), Point(x + xE, y + yE), Scalar(0, 255, 0));
				circle(bgrImg, Point(x + xE, y + yE), 1, Scalar(0, 255, 0), CV_FILLED);
			}
		}
	}
	return bgrImg;
}

//--------------------------------

//Blatat03 - 3.1
double*** compute_HoG(const cv::Mat& img, const int cell_size, std::vector<int>& dims)
{
	int cellsX = int(img.cols / cell_size);
	int cellsY = int(img.rows / cell_size);
	int bins = 9;

	Mat directions = gradients(img);

	//Initalize 3D Array
	double*** HoG = new double**[cellsY];

	for (int j = 0; j < cellsY; j++)
	{
		HoG[j] = new double*[cellsX];

		for (int i = 0; i < cellsX; i++)
		{
			HoG[j][i] = new double[bins];
		}
	}

	//Save Values to dims
	dims.push_back(cellsY);
	dims.push_back(cellsX);
	dims.push_back(bins);
	dims.push_back(cell_size);

	//Aggregate orientations in the HoG
	for (int j = 0; j < cellsY; j++)
	{
		for (int i = 0; i < cellsX; i++)
		{
			for (int n = 0; n < cell_size; n++)
			{
				for (int m = 0; m < cell_size; m++)
				{
					int bin = int(directions.at<float>(j * cell_size + n, i * cell_size + m) * (float(bins) / M_PI));

					if (bin == bins)
					{
						bin--;
					}

					HoG[j][i][bin]++;
				}
			}
		}
	}

	return HoG;
}

Mat gradients(Mat img)
{
	Mat matrix(img.rows, img.cols, CV_32FC1);
	Mat xGradient, yGradient;

	cv::Sobel(img, xGradient, CV_16SC1, 1, 0, 3);
	cv::Sobel(img, yGradient, CV_16SC1, 0, 1, 3);

	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (y != 0 && y != img.rows - 1 && x != 0 && x != img.cols - 1)
			{
				float val = atan2(yGradient.at<short>(y, x), xGradient.at<short>(y, x));

				if (val < 0)
				{
					val += M_PI;
				}

				matrix.at<float>(y, x) = val;
			}
			else
			{
				matrix.at<float>(y, x) = 0;
			}
		}
	}
	return matrix;
}

//--------------------------------

//Blatat03 - 3.2
Mat visualize_HoG(double*** hog, vector<int>& dims)
{
	Mat img(dims[0] * dims[3], dims[1] * dims[3], CV_8UC1, Scalar(0, 0, 0));
	double degree = (M_PI / (dims[2] - 1));
	int t = 10;

	for (int j = 0; j < dims[0]; j++)
	{
		for (int i = 0; i < dims[1]; i++)
		{
			//Find min & max values
			double min = 200;
			double max = -1;

			for (int k = 0; k < dims[2]; k++)
			{
				if (hog[j][i][k] < min)
				{
					min = hog[j][i][k];
				}

				if (hog[j][i][k] > max)
				{
					max = hog[j][i][k];
				}
			}

			for (int b = 0; b < dims[2]; b++)
			{
				double color = (255 - t) / (max - min) * (hog[j][i][b] - max) + 255;
				int length = dims[3] / 2;
				int xE = length * (cos(degree * b) + (degree / 2));
				int yE = length * (sin(degree * b) + (degree / 2));
				int xCenter = i * dims[3] + length;
				int yCenter = j * dims[3] + length;
				line(img, Point(xCenter - xE, yCenter - yE), Point(xCenter + xE, yCenter + yE), Scalar(color));
			}
		}
	}
	return img;
}

//--------------------------------
//Blatt04 - 1
Mat createSimpleTrainingSet(int n, int cols, int rows, int spare, Mat trainingDataMat, float* labels)
{
	Mat Points = Mat(rows, cols, CV_8SC1);
	int mid = rows / 2;
	float randx, randy;

	for (int k = 0; k < n; k++)
	{
		randx = newRandomdoubleGenerator(0, cols - 1);

		if (k < (n / 2))
		{
			labels[k] = 1;
			randy = newRandomdoubleGenerator(0, mid - spare);
			Points.at<schar>(Point(randx, randy)) = 1;
		}
		else
		{
			labels[k] = -1;
			randy = newRandomdoubleGenerator(mid + spare, rows - 1);
			Points.at<schar>(Point(randx, randy)) = -1;
		}

		trainingDataMat.at<float>(k, 0) = randx;
		trainingDataMat.at<float>(k, 1) = randy;
	}

	return Points;
}

Mat drawSet(Mat vals)
{
	Mat img = Mat(vals.rows, vals.cols, CV_8UC3, Scalar(0, 0, 0));

	for (int y = 0; y < vals.rows; y++)
	{
		for (int x = 0; x < vals.cols; x++)
		{
			if (vals.at<schar>(y, x) == 1)
			{
				circle(img, Point(x, y), 3, Scalar(0, 255, 0), CV_FILLED);
			}
			if (vals.at<schar>(y, x) == -1)
			{
				circle(img, Point(x, y), 3, Scalar(0, 0, 255), CV_FILLED);
			}
		}
	}
	return img;
}

Mat trainSVM(int n, int width, int height, int spare, int kernel)
{
	//Setup training data from 1 a)
	Mat trainingDataMat(n, 2, CV_32FC1);
	float* labels = new float[n]();
	Mat img = drawSet(createSimpleTrainingSet(n, width, height, spare, trainingDataMat, labels));
	Mat labelsSVM(n, 1, CV_32FC1, labels);

	//Setup SVM parameters
	CvSVMParams params;
	int iterations = n;

	if (kernel == 1)
	{
		//1 b)
		params.svm_type = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, iterations, 1e-6);
		
		//Train and save SVM
		CvSVM SVM;
		SVM.train_auto(trainingDataMat, labelsSVM, Mat(), Mat(), params);
		SVM.save("SVM");
	}

	return img;
}

Mat testSVM(int width, int height)
{
	Mat result = Mat(height, width, CV_8UC3);
	CvSVM SVM;
	SVM.load("SVM");

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << x , y);
			float response = SVM.predict(sampleMat, true);

			if (response >= 1.0)
				result.at<Vec3b>(y, x) = Vec3b(0, 0, 100);
			else if (response >= 0)
				result.at<Vec3b>(y, x) = Vec3b(0, 0, 150);
			else if (response <= -1.0)
				result.at<Vec3b>(y, x) = Vec3b(0, 100, 0);
			else if (response < 0)
				result.at<Vec3b>(y, x) = Vec3b(0, 150, 0);
		}
	}

	return result;
}

//--------------------------------
