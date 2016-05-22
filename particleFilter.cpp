#include <vector>
#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <limits.h>
#include "particleFilter.h"
#include "utils.h"
static string descriptorVectorFile = "/Users/xinyiguo/SonyCPPGithub/tracker/genfiles/descriptorvector.dat";
using namespace std;
using namespace cv;
double sigmaX = 1;
double sigmaY = 1;
int width = 24;
int height = 48;
double alpha = 0.5;
static double thresholdN = 3; //threshold for neighbor
particleFilter::particleFilter() {
    gsl_rng_env_setup();
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, time(NULL));
    vector<float> descriptorVector = getDescriptorVectorFromFile(descriptorVectorFile);
    HOGDescriptor hog;
    hog.winSize = Size(24, 48);
    hog.blockStride = Size(1,2);
    hog.setSVMDetector(descriptorVector);
    for(int i = 0 ; i < 20; i++){
        availableS.insert(i);
    }
}

particleFilter::~particleFilter() {

}

void particleFilter::addFilters(Mat& frame, vector<Rect> detected, HOGDescriptor& hog){
    for (vector<Rect>::iterator deRec = detected.begin() ; deRec != detected.end(); ++deRec){
        cout << (*deRec).x << endl;
        if(findNeighbor(*deRec)!=-1){
             rectangle(frame, *deRec, Scalar(64, 64, 64) , 2);
             rectangle(frame, filters[findNeighbor(*deRec)].getRecFilter(),Scalar(64, 64, 64) , 2 );
             cout <<"find neighbor is "<<  findNeighbor(*deRec) << endl;
             availableS.erase(findNeighbor(*deRec));
             for(int m = 0; m < filters[findNeighbor(*deRec)].capacity; m++){
             //for(auto p : filters[findNeighbor(*deRec)].particles){
                 filters[findNeighbor(*deRec)].particles[m].xAda = (*deRec).x;
                 filters[findNeighbor(*deRec)].particles[m].yAda = (*deRec).y;
                 cout << "neigbor x is " << filters[findNeighbor(*deRec)].particles[m].x << endl;
             }
             //findNeighbor(*deRec).displayParticles(frame, Scalar(255, 255, 255));
        }
        else if(!availableS.empty()){
                unordered_set<int>::const_iterator it(availableS.begin());
                advance(it,1);
                int addIndex =  *it;
                availableS.erase(addIndex);
                filter * added = &filters[addIndex];
                double prob = filter :: computeSim(hog,  frame, (int)(*deRec).x, (int)(*deRec).y, 2.8);
                cout << "new add probability is " << prob << endl;
                added -> initParticles(*deRec, prob);
                filters[addIndex].displayParticles(frame, Scalar(255, 64, 64));
            }
    }

    for (unordered_set<int>::iterator it = availableS.begin(); it!=availableS.end(); ++it)
        cout << ' ' << *it;

}

void particleFilter :: updateGlobalWeight(Mat& frame){
    double eachFilterSum[20];
    for(int i=0; i<20; i++){
        if(availableS.count(i) == 0) {
            filter* cur = &filters[i];
            for(int j = 0; j < cur->capacity; j++)
                eachFilterSum[i] += cur->particles[j].w;
        }
    }
    double piSum = 0;
    for(int i=0; i<20; i++){
        if(availableS.count(i) == 0 ) {// not avaliable
            piSum += eachFilterSum[i] * filters[i].piPrev;
        }
    }
    cout << "global update sum weight is " << piSum << endl;
    for(int i=0; i<20; i++){
        if(availableS.count(i) == 0 ) {// not avaliable
            double temp = filters[i].piPrev;
            filters[i].piPrev = filters[i].pi;
            filters[i].pi = temp * eachFilterSum[i] /piSum;
            cout << "mixture "<< i << "prev " << temp <<endl;
            cout << "mixture "<< i << "cur weight " << filters[i].particles[0].w << endl;
            cout << "mixture "<< i << "particle weight sum is "<< eachFilterSum[i] << endl;
        }
    }
    for(int i=0; i<20; i++){
        if(availableS.count(i) == 0 ) {// not avaliable
            cout << "mixture pi value for " << i << " is " << filters[i].pi << endl;
        }
    }
}
void particleFilter ::removeFilters(double removeValue, Mat & frame, HOGDescriptor& hog){
    for(int i=0; i<20; i++){
       if(!(availableS.find(i) != availableS.end())){ // not available, means occupied with a filter
            filter cur = filters[i];
            Rect deRec = cur.getRecFilter();
            if(cur.pi < removeValue ){ //|| filter ::computeSim( hog,  frame, deRec.x, deRec.y, 2) < 0 ){
                availableS.insert(i);
                for(int j=0; j<filters[i].capacity; j++){
                    filters[i].particles[j].w = filters[i].particles[j].wNormalized = filters[i].particles[j].wp = 0;
                    filters[i].particles[j].x = filters[i].particles[j].xAda = filters[i].particles[j].xp = 0;
                    filters[i].particles[j].y = filters[i].particles[j].yAda = filters[i].particles[j].yp = 0;
                }
            }
        }
    }
}

void particleFilter :: displayFilters(Mat& frame, Scalar s){
    for(int i=0; i<20; i++){
        if(availableS.count(i) == 0 ){ // not available, means occupied with a filter
            cout << "call each mixture to show" << endl;
            cout <<"weight is "<<filters[i].pi << "prev" << filters[i].piPrev <<  endl;
            filters[i].displayParticles(frame, s);
        }
    }
}

//void mergeFilters();

bool particleFilter :: checkNeighbor(Rect adaP, Rect trackP){
    double dis = sqrt((adaP.x - trackP.x)^2 +( adaP.y - trackP.y)^2);
        if(dis<thresholdN) return true;
    return false;
}

int particleFilter::findNeighbor(Rect adaP){
    int minDis = INT_MAX;
    int res = -1;
    for(int i=0; i<20; i++){
        if(availableS.find(i) == availableS.end()){ // not available, means occupied with a filter
            Rect temp = filters[i].getRecFilter();
            double dis = sqrt((temp.x -adaP.x)^2+(temp.y -adaP.y)^2);
            if( dis < thresholdN && dis < minDis){
                res = i;
                minDis = dis;
            }
        }
    }
    return res;
}

void particleFilter :: process(Mat& frame, vector<Rect> detected, HOGDescriptor& hog){
    cout << "Add filters " << endl;
    addFilters(frame, detected,hog);
    for(int i=0; i<20; i++){
        if(availableS.count(i) == 0){
            filter* cur = &filters[i];
            cur -> transition(alpha);
            cur -> updateWeight(frame, hog, alpha);
            cur -> resample();
        }
    }
    updateGlobalWeight(frame);
    removeFilters(0.009, frame, hog);
    displayFilters(frame, Scalar(64, 255, 64));
}


