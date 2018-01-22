#include "ImageNet.h"

//ImageNet Competition
namespace ImageNet
{
	uchar dataStruct::category = 0;
	map<string, uchar> dataStruct::hashTable;
	map<uchar, string> dataStruct::reverseTable;

	void img::read(const FileNode& node)
	{
		FileNode sourceNode = node["source"];
		FileNode sizeNode = node["size"];
		FileNode objectNode = node["object"];

		folder = (string)node["folder"];
		filename = (string)node["filename"];
		database = (string)sourceNode["database"];
		imgSize.height = (int)sizeNode["height"];
		imgSize.width = (int)sizeNode["width"];

		for (int i = 0; i = objectNode.size(); i++)
		{
			FileNode boxNode = objectNode[i]["bndbox"];
			imgObj tempInfo;

			tempInfo.name = (string)objectNode[i]["name"];
			tempInfo.xMax = (int)boxNode["xmax"];
			tempInfo.xMin = (int)boxNode["xmin"];
			tempInfo.yMax = (int)boxNode["ymax"];
			tempInfo.yMin = (int)boxNode["ymin"];
			object.push_back(tempInfo);
		}
	}

	void img::printData() const
	{
		cout << "---Info of Image---" << endl;
		for (int i = 0; i = object.size(); i++)
		{	
			cout << ">>>obj " << i << endl;
			cout << "folder = " << folder << endl;
			cout << "filename = " << filename << endl;
			cout << "database = " << database << endl;
			cout << "width = " << imgSize.width << endl;
			cout << "height = " << imgSize.height << endl;
			cout << "name = " << object[i].name << endl;
			cout << "xMax = " << object[i].xMax << endl;
			cout << "xMin = " << object[i].xMin << endl;
			cout << "yMax = " << object[i].yMax << endl;
			cout << "yMin = " << object[i].yMin << endl << endl;
		}
		cout << "-----end print-----" << endl;
	}

	Point img::readPoint_ROI(int index) const
	{
		Point pointROI;
		pointROI.x = this->object[index].xMin;
		pointROI.y = this->object[index].yMin;
		return pointROI;
	}

	Size img::readSize_ROI(int index) const
	{
		Size sizeROI;
		sizeROI.width = this->object[index].xMax - this->object[index].xMin;
		sizeROI.height = this->object[index].yMax - this->object[index].xMin;
		return sizeROI;
	}

	static ostream& operator<<(ostream& out, const img& data)
	{
		data.printData();
		return out;
	}
	
	template<typename T>
	static ostream& operator<<(ostream& out, const vector<T>& data)
	{
		for (int i = 0; i < data.size(); i++)
		{
			out << data[i] << " ";
		}
		out << endl;
		return out;
	}

	template<>
	static ostream& operator<<(ostream& out, const vector<uchar>& data)
	{
		for (int i = 0; i < data.size(); i++)
		{
			out << static_cast<int>(data[i]) << " ";
		}
		out << endl;
		return out;
	}

	void loadXML(string& xmlFolder,	vector<img>& ImgInfo)
	{
		vector<string> xmlFilePath;
		
		ImgNet_ML::readImgNamefromFile(xmlFolder, xmlFilePath);
		for (int i = 0; i < xmlFilePath.size(); i++)
		{
			FileStorage fs(xmlFilePath[i], FileStorage::READ);
			if (!fs.isOpened()) // failed
			{
				cout << "Open File Failed!" << endl;
				return;
			}
			FileNode rootNode = fs["annotation"];

			img imgData;
			imgData.read(rootNode);
			ImgInfo.push_back(imgData);

			fs.release();
		}
	}

	void reWriteXML(string& xmlFolder)
	{
		vector<string> xmlFilePath;
		ImgNet_ML::readImgNamefromFile(xmlFolder, xmlFilePath);

		for (int i = 0; i < xmlFilePath.size(); i++)
		{
			vector<string> contentXML;
			readXML(xmlFilePath[i], contentXML);
			if (contentXML[0] == R"(<?xml version="1.0"?>)")
			{
				return;
			}

			ofstream oData(xmlFilePath[i], ios::out);
			oData << R"(<?xml version="1.0"?>)" << endl;
			oData << R"(<opencv_storage>)" << endl;

			for (int i = 0; i < contentXML.size(); i++)
			{
				oData << contentXML[i] << endl;
			}

			oData << R"(</opencv_storage>)" << endl;
			oData.close();
		}
	}

	void readXML(string& path, vector<string>& data)
	{
		ifstream iData(path, ios::in);
		string line;

		while (iData.peek() != EOF && getline(iData, line))
		{
			data.push_back(line);
		}
		iData.close();
	}

	void extractImg(vector<img>& imgInfo, vector<string>& imgFolderPath, vector<vector<uchar>>& lable, vector<Mat>& img_ROI)
	{
		vector<Mat> imgData;
		for (int i = 0; i < imgFolderPath.size(); i++)
		{
			vector<string> imgPath;
			ImgNet_ML::readImgNamefromFile(imgFolderPath[i], imgPath);
			readImg(imgPath, imgData);
		}

		if (imgData.size() != imgInfo.size())
		{
			cerr << "Num of XML file is not equal to num of image!" << endl;
			return;
		}

		for (int i = 0; i < imgInfo.size(); i++)
		{
			vector<Mat> processedMat = morphology_Preprocess(imgData[i]);
			vector<Mat> temp_ROI;
			showImg(processedMat[0]);

			for (int j = 0; j < imgInfo[i].objSize(); j++)
			{
				Point pointROI = imgInfo[i].readPoint_ROI(j);
				Size sizeROI = imgInfo[i].readSize_ROI(j);
				Mat image_ROI = ImgNet_ML::regionExtraction(imgData[i], pointROI.x, pointROI.y, sizeROI.width, sizeROI.height, false);
				img_ROI.push_back(image_ROI);
				temp_ROI.push_back(image_ROI);
			}
			encodeLable(imgInfo[i], lable);
			transferMat(processedMat, img_ROI, temp_ROI, lable);
		}
	}

	void initLable(const vector<img>& imgInfo)
	{
		set<string> lableSet;
		for (int i = 0; i < imgInfo.size(); i++)
		{
			vector<imgObj> tempObj = imgInfo[i].readObj();
			for (int j = 0; j < tempObj.size(); j++)
			{
				lableSet.insert(tempObj[j].name);
			}
		}

		vector<string> lableName;
		for (auto iter = lableSet.begin(); iter != lableSet.end(); iter++)
		{
			lableName.push_back(*iter);
		}
		dataStruct::category = lableName.size() + 1;

		dataStruct::hashTable[lableName[0]] = static_cast<uchar>(dataStruct::imgLable::n00007846);
		dataStruct::hashTable[lableName[1]] = static_cast<uchar>(dataStruct::imgLable::n00477639);
		dataStruct::hashTable[lableName[2]] = static_cast<uchar>(dataStruct::imgLable::n01443537);
		dataStruct::hashTable[lableName[3]] = static_cast<uchar>(dataStruct::imgLable::n01495701);
		dataStruct::hashTable[lableName[4]] = static_cast<uchar>(dataStruct::imgLable::n01503061);
		dataStruct::hashTable[lableName[5]] = static_cast<uchar>(dataStruct::imgLable::n01514668);
		dataStruct::hashTable[lableName[6]] = static_cast<uchar>(dataStruct::imgLable::other);

		dataStruct::reverseTable[static_cast<uchar>(dataStruct::imgLable::n00007846)] = "person";
		dataStruct::reverseTable[static_cast<uchar>(dataStruct::imgLable::n00477639)] = "horseback";
		dataStruct::reverseTable[static_cast<uchar>(dataStruct::imgLable::n01443537)] = "goldfish";
		dataStruct::reverseTable[static_cast<uchar>(dataStruct::imgLable::n01495701)] = "ray";
		dataStruct::reverseTable[static_cast<uchar>(dataStruct::imgLable::n01503061)] = "bird";
		dataStruct::reverseTable[static_cast<uchar>(dataStruct::imgLable::n01514668)] = "cock";
		dataStruct::reverseTable[static_cast<uchar>(dataStruct::imgLable::other)] = "other";
	}

	void encodeLable(img& imgInfo, vector<vector<uchar>>& Y)
	{
		vector<imgObj> tempObj = imgInfo.readObj();
		for (int j = 0; j < tempObj.size(); j++)
		{
			vector<uchar> imgLable(dataStruct::category, 0);
			for (int k = 0; k < dataStruct::category; k++)
			{
				uchar lable = dataStruct::hashTable[tempObj[j].name];
				if (k == lable || k == dataStruct::category - 1)
				{
					imgLable.push_back(1);
					break;
				}
			}
			Y.push_back(imgLable);
		}
	}

	bool isSimilar(Mat& srcImage1, Mat& srcImage2)
	{
		if (srcImage1.empty() || srcImage2.empty())
		{
			return false;
		}
		int histSize = 256;
		int channels = { 0 };
		float hist_ranges[] = { 0, 255 };
		const float* rangePtr = { hist_ranges };
		Mat hist_source, hist_target;

		calcHist(&srcImage1, 1, &channels, Mat(), hist_source, 1, &histSize, &rangePtr);
		normalize(hist_source, hist_source, 0, 1, NORM_MINMAX, -1, Mat());

		calcHist(&srcImage2, 1, &channels, Mat(), hist_target, 1, &histSize, &rangePtr);
		normalize(hist_source, hist_source, 0, 1, NORM_MINMAX, -1, Mat());

		double similarity = compareHist(hist_source, hist_target, HISTCMP_CORREL);

		return similarity > 0.4 ? true : false;
	}

	vector<Mat> morphology_Preprocess(Mat& srcGray)
	{
		Mat result;
		vector<Mat> target;

		morphologyEx(srcGray, result, MORPH_GRADIENT, Mat(1, 2, CV_8U, Scalar(1)));
		threshold(result, result, 255 * (0.1), 255, THRESH_BINARY);
		morphologyEx(result, result, MORPH_CLOSE, Mat(5, 20, CV_8U, Scalar(1)));

		vector<vector<Point>> target_contours;
		findContours(result.clone(), target_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		for (int i = 0; i < target_contours.size(); i++)
		{
			Rect rect = boundingRect(target_contours[i]);
			double wh_ratio = double(rect.width) / rect.height;
			int sub = countNonZero(result(rect));
			double ratio = double(sub) / rect.area();

			if (wh_ratio > 0.1 && wh_ratio < 10 && rect.height > 30 && rect.width > 30 && ratio > 0.2)
			{
				target.push_back(srcGray(rect));
			}
		}
		return target;
	}

	void transferMat(vector<Mat>& source, vector<Mat>& target, vector<Mat>& image_ROI, vector<vector<uchar>>& lable)
	{
		for (int i = 0; i < source.size(); i++)
		{
			for (int j = 0; j < image_ROI.size(); j++)
			{
				if (!isSimilar(source[i], image_ROI[j]))
				{
					target.push_back(move(source[i]));
					vector<uchar> other_Lable(dataStruct::category, 0);
					other_Lable[dataStruct::category - 1] = 1;
					lable.push_back(other_Lable);
				}
			}
		}
	}

	void readImg(vector<string>& imgPath, vector<Mat>& imgData)
	{
		for (int i = 0; i < imgPath.size(); i++)
		{
			Mat img = imread(imgPath[i], CV_LOAD_IMAGE_GRAYSCALE);
			imgData.push_back(img);
		}
	}

	void OTSU_Mask(Mat& img_ROI, Mat& mask)
	{
		if (img_ROI.channels() != 1)
		{
			cvtColor(img_ROI, img_ROI, CV_RGB2GRAY);
		}
		
		Mat threMat;
		threshold(img_ROI, threMat, 150, 255, THRESH_OTSU | THRESH_BINARY_INV);
		mask = threMat;
	}

	void extractLBP(vector<Mat>& img_ROI, vector<vector<uchar>>& LBP_img)
	{
		for (int i = 0; i < img_ROI.size(); i++)
		{
			Mat mask, targetImg, LBPmat, poolLBP;
			OTSU_Mask(img_ROI[i], mask);

			img_ROI[i].copyTo(targetImg, mask);
			resize(targetImg, targetImg, Size(28, 28));
			LBPmat = ImgNet_ML::OLBP(targetImg);
			poolLBP = poolFeature(LBPmat).reshape(0, 1);
			const uchar* featureVec = poolLBP.data;
			vector<uchar> LBP_feature(featureVec, featureVec + poolLBP.cols);

			LBP_img.push_back(move(LBP_feature));
		}
	}

	Mat poolFeature(Mat& feature, int cellSize)
	{
		Mat poolMat(feature.size() / cellSize, feature.type());
		int nRows = feature.rows;
		int nCols = feature.cols;

		for (int y = 0; y < nRows - 1; y+=2)
		{
			for (int x = 0; x < nCols - 1; x+=2)
			{
				vector<uchar> cell(4, 0);
				cell[0] = feature.at<uchar>(y, x);
				cell[1] = feature.at<uchar>(y + 1, x);
				cell[2] = feature.at<uchar>(y + 1, x + 1);
				cell[3] = feature.at<uchar>(y, x + 1);

				poolMat.at<uchar>(y/cellSize, x/cellSize) = maxPix(cell);
			}
		}
		return poolMat;
	}

	uchar maxPix(vector<uchar>& pixels)
	{
		uchar result = 0;
		for (int i = 0; i < pixels.size(); i++)
		{
			std::max(result, pixels[i]);
		}

		return result;
	}

	ImgNet_ML::neuron_network<uchar> trainWithNN(int numOfNeuron, int depthNN)
	{
		string xmlFolder{ R"(C:\Users\Acer\Desktop\work\train\train_info)" };
		vector<img> train_ImgInfo;
		readXML(xmlFolder, train_ImgInfo);
		initLable(train_ImgInfo);

		// train_img extraction
		string imgFolder{ R"(C:\Users\Acer\Desktop\work\train\train_img)" };
		vector<Mat> img_ROI;
		vector<vector<uchar>> train_Lable;
		img_Extraction(imgFolder, train_ImgInfo, img_ROI, train_Lable);

		// validate_img extraction
		string validateXmlFolder{ R"(C:\Users\Acer\Desktop\work\validate\val_info)" };
		vector<img> validate_ImgInfo;
		readXML(validateXmlFolder, validate_ImgInfo);

		string validateImgFolder{ R"(C:\Users\Acer\Desktop\work\validate\val_img)" };
		vector<Mat> validate_ROI;
		vector<vector<uchar>> validate_Lable;
		img_Extraction(validateImgFolder, validate_ImgInfo, validate_ROI, validate_Lable);

		// feature extraction of image ROI (28*28)
		vector<vector<uchar>> LBP_trainImg, LBP_validate;
		extractLBP(img_ROI, LBP_trainImg);
		extractLBP(validate_ROI, LBP_validate);

		ImgNet_ML::neuron_network<uchar> NN(LBP_trainImg.size(), train_Lable[0].size(), numOfNeuron, depthNN, 1, ImgNet_ML::activation_for_hyperbolic);
		NN.train(LBP_trainImg, train_Lable);
		NN.predict(LBP_validate, validate_Lable);
		return NN;
	}
	
	void showImg(Mat& image)
	{
		namedWindow("imgWindow", CV_WINDOW_AUTOSIZE);
		imshow("imgWindow", image);
		waitKey(0);
	}

	void DemoImgNet()
	{
		// train 10-MLs NN
		ImgNet_ML::neuron_network<uchar> NN = trainWithNN(200, 6);

		// classify the test images
		vector<vector<uchar>> LBP_testImg, test_Lable;
		make_TestData(LBP_testImg, test_Lable);
		NN.predict(LBP_testImg, test_Lable);
	}

	void readXML(string xmlFolder, vector<img>& ImgInfo)
	{
		// XML info loading
		vector<string> folderPath;
		ImgNet_ML::readImgNamefromFile(xmlFolder, folderPath);
		for (int i = 0; i < folderPath.size(); i++)
		{
			reWriteXML(folderPath[i]);
		}

		for (int i = 1; i < folderPath.size(); i++)
		{
			loadXML(folderPath[i], ImgInfo);
		}
	}

	void img_Extraction(string imgFolder, vector<img>& train_ImgInfo, vector<Mat>& img_ROI, vector<vector<uchar>>& train_Lable)
	{
		vector<string> imgFolderPath;
		ImgNet_ML::readImgNamefromFile(imgFolder, imgFolderPath);
		extractImg(train_ImgInfo, imgFolderPath, train_Lable, img_ROI);
	}

	void make_TestData(vector<vector<uchar>>& LBP_testImg, vector<vector<uchar>>& test_Lable)
	{
		// read the test images
		string testXmlFolder{ R"(C:\Users\Acer\Desktop\work\test\test_info)" };
		vector<img> test_ImgInfo;
		readXML(testXmlFolder, test_ImgInfo);

		string testImgFolder{ R"(C:\Users\Acer\Desktop\work\test\test_img)" };
		vector<Mat> test_ROI;
		img_Extraction(testImgFolder, test_ImgInfo, test_ROI, test_Lable);

		// feature extraction of image ROI (28*28)
		extractLBP(test_ROI, LBP_testImg);
	}
}