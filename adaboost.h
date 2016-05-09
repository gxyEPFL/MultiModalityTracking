#ifndef ADABOOST_H
#define ADABOOST_H

#endif // ADABOOST_H
#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml/ml.hpp>

vector<float>getDescriptorVectorFromFile(string fileName);
void processVideoHOG(char* videoFilename,  BackgroundSubtractorMOG& pMOG, HOGDescriptor hog);
int detector(char* videoFilename);
vector<Rect> getDetected(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData, Mat& original);
