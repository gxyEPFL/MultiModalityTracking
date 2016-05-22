#ifndef PARTICLE_H
#define PARTICLE_H
#endif // PARTICLE_H
#include <vector>
#include <unordered_set>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "filter.h"
using namespace std;
using namespace cv;
class particleFilter {
public:
    unordered_set<int> availableS;
    int nFilters;
    gsl_rng* rng;
    filter filters[20];
    particleFilter();
    ~particleFilter();
    void updateGlobalWeight(Mat& frame);
    void displayFilters(Mat& frame, Scalar s);
    void removeFilters(double removeValue, Mat & frame, HOGDescriptor& hog);
    //void mergeFilters();
    void addFilters(Mat& frame, vector<Rect> detected, HOGDescriptor& hog);
    void process(Mat& frame,vector<Rect> detected, HOGDescriptor& hog);
    int findNeighbor(Rect adaP);
    bool checkNeighbor(Rect adaP, Rect trackP);
};
