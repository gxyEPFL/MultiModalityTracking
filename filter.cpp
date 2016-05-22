#include "filter.h"
#include "utils.h"
#include <math.h>
#include <vector>
#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <gsl/gsl_rng.h>
static string descriptorVectorFile = "/Users/xinyiguo/SonyCPPGithub/tracker/genfiles/descriptorvector.dat";
using namespace std;
using namespace cv;

filter::filter() {
    gsl_rng_env_setup();
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, time(NULL));
    vector<float> descriptorVector = getDescriptorVectorFromFile(descriptorVectorFile);
    HOGDescriptor hog;
    hog.winSize = Size(24, 48);
    hog.blockStride = Size(1,2);
    hog.setSVMDetector(descriptorVector);
}

filter::~filter() {

}

/* for each detection region, add Nparcitles with weight 0, the prob for this mixture is the weight from detector directly
need normalized globally*/
void filter::initParticles(Rect region, double prob) {
    cout << "initial particle's at here " <<  region.x;
    for (int i=0; i < capacity; i++) {
        particles[i].xAda = particles[i].xp = particles[i].x = region.x; //left top of the target
        particles[i].yAda = particles[i].yp = particles[i].y = region.y; //left top of the target
        particles[i].w = particles[i].wNormalized = particles[i].wp = (double)1/capacity;

        }
    pi = piPrev = prob;
}

/* for each detection, sample from the proposal distribution, weights excluded*/
/* @parameter x,y : the detector position for this mixture (data association)    */
 void filter::transition(double alpha){ //  w and h are the max coodinate value for each axis
     for(int i=0; i<capacity; i++){
        particle pi = particles[i];
        particle pn;
        double x0 = pi.xAda + 3 * gsl_ran_gaussian(rng, sigmaX*sigmaX);
        double y0 = pi.yAda + 2 * gsl_ran_gaussian(rng, sigmaY*sigmaY);
        double x1 =  pi.x +  3 * gsl_ran_gaussian(rng, sigmaX*sigmaX);
        double y1 =  pi.y +  2 * gsl_ran_gaussian(rng, sigmaY*sigmaY);
        pn.x = (float) MAX( 0.0, MIN((float) w - width, alpha * x1 + (1-alpha) * x0));
        pn.y = (float) MAX( 0.0, MIN((float) h - height, alpha * y1 + (1-alpha) * y0));
        pn.xp = pi.x;
        pn.yp = pi.y;
        pn.w = pi.w;
        pn.wNormalized = pi.wNormalized;
        pn.wp = pi.wp;
        particles[i] = pn;
     }
 }

 /*update transated particle weight based on similarities with the previous detected hog discriptor*/
 void filter::updateWeight(Mat& frame,  const HOGDescriptor& hog, double alpha){
     double eachFilterSum = 0;
     for(int i = 0; i < capacity; i++){
            double temp = particles[i].wp;
            particles[i].wp = particles[i].wNormalized;
            if(proposalP(hog, frame, &particles[i], 0.5, 0)!=0){
                particles[i].w =  computeSim(hog, frame, particles[i].x, particles[i].y, 2.8)
                         * dynamic_prob(&particles[i])
                         /proposalP(hog, frame, &particles[i], 0.5, 0);
                if(particles[i].w == 0){
                cout << "particle previous " << temp << endl;
                cout << "particle appearnce  "<< computeSim(hog, frame, particles[i].x, particles[i].y, 0)<< endl;
                cout << "dynamic "<< dynamic_prob(&particles[i]) << endl;
                cout << "position "<<particles[i].x << endl;
                   }
                }
            else particles[i].w = 0;
            eachFilterSum += particles[i].w;
    }
    cout << "mixture particle weight sum is "<< eachFilterSum  << endl;

     for(int i=0; i < capacity; i++){
            particles[i].wNormalized = particles[i].w / eachFilterSum;
        }
 }

/*resample the particles based on the weight, use wheel round method */
void filter::resample(){
    particle * newParticles;
    newParticles = (particle*) malloc(capacity * sizeof(particle));
    double beta = 0.0;
    double maxWeight =0;
    for(int i=0; i<capacity; i++){
        maxWeight = max(maxWeight,  particles[i].wNormalized);
    }
    int index =(int)gsl_rng_uniform (rng)*capacity;
    for(int k=0 ; k < capacity; k++){
        beta += gsl_rng_uniform (rng) * 2 * maxWeight;
        while(beta > particles[index].wNormalized){
            beta -= particles[index].wNormalized;
            index = (index+1) %capacity;
        }
        newParticles[k] = particles[index];
    }

    for(int i=0; i<capacity; i++){
        particles[i] = newParticles[i];
       // particles[i].w =particles[i].wNormalized= 1/capacity;
    }
    free(newParticles);
    //cout << "Resample finished partly" << endl;
}

Rect filter::getRecFilter(){
     double cX=0, cY=0;
     for(int k=0 ; k < capacity; k++){
        cX += particles[k].x;
        cY += particles[k].y;
     }
     return Rect(cX/capacity, cY/capacity, 24, 48);
}

//http://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#gsc.tab=0
/*compute sim based on the exsiting hog descriptor and detector apperance model*/
double filter::computeSim(const HOGDescriptor& hog, Mat& frame, int x, int y, double hitThreshold){
        vector<Point> searchLoc;
        vector<Point> foundLoc;
        vector<double> weights;
        searchLoc.push_back(Point((double)x, (double)y));
        hog.detect(frame, foundLoc, weights, 0, Size(24,48), Size(1,1), searchLoc);
        if(weights.empty())
            return 0;
        return MIN(1,(MAX(0,weights.at(0)-2)));
}

/*dynamic model*/
double filter::dynamic_prob(particle *p){
    double xProbability = gsl_ran_gaussian_pdf ((p->x - p->xp)/3.0, sigmaX);
    double yProbability = gsl_ran_gaussian_pdf ((p->y - p->yp)/2.0, sigmaY);
return sqrt(xProbability*yProbability);
}

/*proposal distribution*/
double filter::proposalP(const HOGDescriptor& hog, Mat& frame, particle *p, double alpha, double hitThreshold){
    double xProbability = gsl_ran_gaussian_pdf ((p->x - p->xAda)/3.0, sigmaX);
    double yProbability = gsl_ran_gaussian_pdf ((p->y - p->yAda)/2.0, sigmaY);
    double apperance =  sqrt(xProbability*yProbability);
    double dynamic = dynamic_prob(p);
return alpha * apperance + (1-alpha)*dynamic;
}


void filter::displayParticles(Mat& frame, Scalar s){ //Scalar(64, 255, 64)
    cout << "call the display";
    for(int i=0; i<capacity; i++){
            int x0 = cvRound(particles[i].x);
            int y0 = cvRound(particles[i].y);
            Rect r = Rect(min(720-width, x0), min(480-height,y0), width, height);
            rectangle(frame, r, s , 2);
    }
    imshow("result",frame);
    waitKey(1000);
}
