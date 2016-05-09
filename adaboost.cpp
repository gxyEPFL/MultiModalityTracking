#include "bsOpenCV.h"
#include "descriptor.h"
#include "adaboost.h"
#include "utils.h"
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
static string descriptorVectorFile = "/Users/xinyiguo/SonyCPPGithub/trainHOG/genfiles/descriptorvector.dat";
using namespace std;

void showDetections(const vector<Rect>& found, Mat& imageData, Mat& original) {
    vector<Rect> found_filtered;
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Rect r = found[i];
        for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    for (i = 0; i < found_filtered.size(); i++) {
        Rect r = found_filtered[i];
        rectangle(original, r, Scalar(64, 255, 64), 1);
    }
}

void detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData, Mat& original) {
    vector<Rect> found;
    Size padding(Size(2, 2));
    Size winStride(Size(1, 1));
    vector<double> foundWeights;
    /*void HOGDescriptor::detectMultiScale(const Mat& img, vector<Rect>& foundLocations, vector<double>& foundWeights,
    double hitThreshold, Size winStride, Size padding, double scale0, double finalThreshold, bool useMeanshiftGrouping)*/
    hog.detectMultiScale(imageData, found, foundWeights, hitThreshold, winStride, padding, 0.8, 2);
   /* for(int i=0; i<foundWeights.size(); i++){
        cout << "found weight is" << foundWeights.at(i)<< endl;
        cout << "found position is" << found.at(i) << endl;
    }*/
    showDetections(found, imageData, original);
}

vector<Rect> getDetected(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData, Mat& original){
    vector<Rect> found;
    Size padding(Size(2, 2));
    Size winStride(Size(1, 1));
    vector<double> foundWeights;
    /*void HOGDescriptor::detectMultiScale(const Mat& img, vector<Rect>& foundLocations, vector<double>& foundWeights,
    double hitThreshold, Size winStride, Size padding, double scale0, double finalThreshold, bool useMeanshiftGrouping)*/
    hog.detectMultiScale(imageData, found, foundWeights, hitThreshold, winStride, padding, 0.8, 2);
    return found;
}

/*void processVideo(HOGDescriptor hog, double hitThreshold) {
    Mat frame;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir ("/Users/xinyiguo/Desktop/VOC2012/Sequence10")) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            if (ent -> d_name[0] == '.') continue;
            string s1 = "/Users/xinyiguo/Desktop/VOC2012/Sequence10/";
            string s2 = "/Users/xinyiguo/Desktop/VOC2012/original10/";
            string s = s1 + ent -> d_name;
            string name = ent -> d_name;
            string o = s2 + ent -> d_name;
            printf ("%s\n", s.c_str());
            frame = imread(s);
            Mat original = imread(o);
            imshow("load original image", frame);
            waitKey(1);
            detectTest(hog, 2.8, frame, original);
            imshow("HOG custom detection", original);
            imwrite( "/Users/xinyiguo/Desktop/VOC2012/result10/" + name, original);
            waitKey(1);
        }
    closedir (dir);
    }
} */

void processVideoHOG(char* videoFilename,  BackgroundSubtractorMOG& pMOG, HOGDescriptor hog) {
    VideoCapture capture(videoFilename);
    Mat frame;
    int keyboard=0;
    if(!capture.isOpened()){
        cerr << "Unable to open video file: " << videoFilename << endl;
        exit(EXIT_FAILURE);
    }
    while((char)keyboard != 'q' && (char)keyboard != 27){
        if(!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }
        Mat bsRes = pMOG.processFrame( frame);
        detectTest(hog, 2.8, bsRes, frame);
        imshow("HOG custom detection", frame);
        keyboard = waitKey( 1 );
    }
    capture.release();
}

int detector(char* videoFilename) {
    vector<float> descriptorVector = getDescriptorVectorFromFile(descriptorVectorFile);
    BackgroundSubtractorMOG pMOG = BackgroundSubtractorMOG();
    HOGDescriptor hog;
    hog.winSize = Size(24, 48);
    std::cout << hog.blockSize << endl;
    hog.blockStride = Size(1,2);
    hog.setSVMDetector(descriptorVector);
    processVideoHOG(videoFilename, pMOG, hog);
    return 0;
}

