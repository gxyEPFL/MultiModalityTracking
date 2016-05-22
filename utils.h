#ifndef UTILS_H
#define UTILS_H

#endif // UTILS_H
#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml/ml.hpp>
using namespace std;
vector<float> getDescriptorVectorFromFile(string fileName);
