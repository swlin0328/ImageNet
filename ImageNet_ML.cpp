#include "ImageNet_ML.h"

//ºÊ·þ¦¡¾Ç²ß
namespace ImgNet_ML
{
	Mat OLBP(Mat& srcImage)
	{
		int nRows = srcImage.rows;
		int nCols = srcImage.cols;
		Mat gray = srcImage;
		if (srcImage.channels() != 1)
		{
			cvtColor(srcImage, gray, COLOR_BGR2GRAY);
		}

		Mat resultMat(gray.size(), gray.type());
		for (int y = 1; y < nRows - 1; y++)
		{
			for (int x = 1; x < nCols - 1; x++)
			{
				uchar neighbor[8] = { 0 };
				neighbor[0] = gray.at<uchar>(y - 1, x - 1);
				neighbor[1] = gray.at<uchar>(y - 1, x);
				neighbor[2] = gray.at<uchar>(y - 1, x + 1);
				neighbor[3] = gray.at<uchar>(y, x + 1);
				neighbor[4] = gray.at<uchar>(y + 1, x + 1);
				neighbor[5] = gray.at<uchar>(y + 1, x);
				neighbor[6] = gray.at<uchar>(y + 1, x - 1);
				neighbor[7] = gray.at<uchar>(y, x - 1);

				uchar center = gray.at<uchar>(y, x);
				uchar temp = 0;
				for (int k = 0; k < 8; k++)
				{
					temp += ((neighbor[k] > center) * (1 << k));
				}
				resultMat.at<uchar>(y, x) = temp;
			}
		}
		return resultMat;
	}

	Mat regionExtraction(Mat& srcImage, int xRoi, int yRoi, int widthRoi, int heightRoi, bool open)
	{
		Mat roiImage(srcImage.rows, srcImage.cols, CV_8UC3);
		srcImage(Rect(xRoi, yRoi, widthRoi, heightRoi)).copyTo(roiImage);

		if (open == true)
		{
			namedWindow("roiImage", CV_WINDOW_AUTOSIZE);
			imshow("roiImage", roiImage);
			waitKey(0);
		}
		return roiImage;
	}

	void readImgNamefromFile(string folderName, vector<string>& imgPaths)
	{
		imgPaths.clear();
		WIN32_FIND_DATA file;
		int i = 0;
		char tempFilePath[MAX_PATH + 1];

		sprintf_s(tempFilePath, "%s/*", folderName.c_str());
		HANDLE handle = FindFirstFile(tempFilePath, &file);
		FindNextFile(handle, &file);
		FindNextFile(handle, &file);
		if (handle != INVALID_HANDLE_VALUE)
		{
			do
			{
				sprintf_s(tempFilePath, "%s\\", folderName.c_str());
				imgPaths.push_back(file.cFileName);
				imgPaths[i].insert(0, tempFilePath);
				i++;
			} while (FindNextFile(handle, &file));
		}
		FindClose(handle);
	}

	double activation_for_hyperbolic(double wX)
	{
		return tanh(wX);
	}

	double square_error(vector<vector<uchar>>& X, vector<uchar>& Y, vector<double>& w, const function<double(double)>& actF)
	{
		double errValue = 0;
		for (int i = 0; i < X.size(); i++)
		{
			errValue += pow((Y[i] - actF(dot(X[i], w))), 2);
		}
		return errValue;
	}
}