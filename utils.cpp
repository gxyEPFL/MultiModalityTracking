#include <utils.h>
#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml/ml.hpp>
vector<float> getDescriptorVectorFromFile(string fileName){
    vector<float> res;
    string separator = " "; // Use blank as default separator between single features
    ifstream myfile(fileName.c_str());
    string line;
    while (getline(myfile,line)){
        stringstream  lineStream(line);
        float value;
        while(lineStream >> value)
        {
            res.push_back(value);
        }
    }
    myfile.close();
    return res;
}
