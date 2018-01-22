#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <casserT>
#include <map>
#include <set>
#include <iostream>

#define __ImageNet

#pragma once
#ifndef __MLlib
#define __MLlib
#include "ImageNet_ML.h"
#endif

#pragma once
#ifndef __Mathlib
#define __Mathlib
#include "ImageNet_Math.h"
#endif

using namespace std;
//ImageNet Competition
namespace ImageNet
{
	void DemoImgNet();

	static struct dataStruct
	{
		static map<string, uchar> hashTable;
		static map<uchar, string> reverseTable;
		static uchar category;
		static enum class imgLable : uchar { n00007846 = 0, n00477639, n01443537, n01495701, n01503061, n01514668, other };
	};

	struct imgObj
	{
		string name;
		int xMin;
		int xMax;
		int yMin;
		int yMax;
	};

	class img
	{
	public:
		void read(const FileNode& node);
		void printData() const;
		Point readPoint_ROI(int index) const;
		Size readSize_ROI(int index) const;
		int objSize() { return object.size(); }
		vector<imgObj> readObj() const { return object; };

	private:
		mutable string folder;
		mutable string filename;
		mutable string database;
		mutable vector<imgObj> object;

		mutable struct { int width; int height; } imgSize;
	};

	void initLable(const vector<img>& imgInfo);
	
	void loadXML(string& xmlFolder, vector<img>& ImgInfo);

	void reWriteXML(string& xmlFolder);

	void readXML(string& path, vector<string>& data);

	void readImg(vector<string>& imgPath, vector<Mat>& imgData);

	void encodeLable(img& imgInfo, vector<vector<uchar>>& Y);

	vector<Mat> morphology_Preprocess(Mat& srcImg);

	bool isSimilar(Mat& srcImage1, Mat& srcImage2);

	void extractImg(vector<img>& imgInfo, vector<string>& imgFolderPath, vector<vector<uchar>>& lable, vector<Mat>& img_ROI);

	void OTSU_Mask(Mat& img_ROI, Mat& mask);

	void extractLBP(vector<Mat>& img_ROI, vector<vector<uchar>>& LBP_img);

	Mat poolFeature(Mat& feature, int cellSize = 2);

	uchar maxPix(vector<uchar>& pixels);

	void transferMat(vector<Mat>& source, vector<Mat>& target, vector<Mat>& image_ROI, vector<vector<uchar>>& lable);

	void showImg(Mat& image);

	void DemoImgNet();

	void readXML(string xmlFolder, vector<img>& ImgInfo);

	void img_Extraction(string imgFolder, vector<img>& train_ImgInfo, vector<Mat>& img_ROI, vector<vector<uchar>>& train_Lable);

	void make_TestData(vector<vector<uchar>>& LBP_testImg, vector<vector<uchar>>& test_Lable);

	ImgNet_ML::neuron_network<uchar> trainWithNN(int numOfNeuron, int depthNN);
}