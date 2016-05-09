#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H
#endif // DESCRIPTOR_H
#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;
class detector
{
public:
    detector();
    ~detector();
    void saveDescriptorVectorToFile(vector<float>& descriptorVector, vector<unsigned int>& _vectorIndices, string fileName);
    void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions);
    void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog);
    //void showDetections(const vector<Rect>& found, Mat& imageData, Mat& original);
    //void detectTrainingSetTest(const HOGDescriptor& hog, const double hitThreshold, const vector<string>& posFileNames, const vector<string>& negFileNames);
    int detection();
};
