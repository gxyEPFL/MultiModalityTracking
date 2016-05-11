#include "particleFilter.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <gsl/gsl_rng.h>
#include <vector>
#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <opencv2/ml/ml.hpp>
#include <math.h>
static string descriptorVectorFile = "/Users/xinyiguo/SonyCPPGithub/tracker/genfiles/descriptorvector.dat";
using namespace std;
using namespace cv;
double sigmaX = 1;
double sigmaY = 1;
int width = 24;
int height = 48;
static int removefLAG = 100;
particleFilter::particleFilter() {
    totalParticles = 180;
    gsl_rng_env_setup();
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, time(NULL));
    vector<CvRect> region;
    vector<float> descriptorVector = getDescriptorVectorFromFile(descriptorVectorFile);
    HOGDescriptor hog;
    hog.winSize = Size(24, 48);
    hog.blockStride = Size(1,2);
    hog.setSVMDetector(descriptorVector);
}

particleFilter::~particleFilter() {

}

/* for each detection in initial frame, add Nparcitles with weight 0 in the place with origial weight */
void particleFilter::initParticles(vector<Rect> region,  gsl_rng* rng) {
    region.erase (region.begin(),region.begin()+2);
    int orgTarget = region.size();
    cout<<"original target" << orgTarget << endl;
    nFilters = 9;
    int eachFilter = totalParticles/nFilters;
    for (int i=0; i < totalParticles; i++) {
        int j= i/eachFilter;
        particles[i].xp = particles[i].x = region.at(j % orgTarget).x + gsl_ran_gaussian(rng, sigmaX*sigmaX); //left top of the target
        particles[i].yp = particles[i].y = region.at(j % orgTarget).y + gsl_ran_gaussian(rng, sigmaX*sigmaX); //left top of the target
        particles[i].velocityX = particles[i].velocityXp = 1 +  gsl_ran_gaussian(rng, sigmaX*sigmaX); // maybe wrong
        particles[i].velocityY = particles[i].velocityYp = 1 +  gsl_ran_gaussian(rng, sigmaX*sigmaX);
        particles[i].wNormalized = (double)1/(totalParticles/orgTarget);
        particles[i].id = j;
        particles[i].w = particles[i].wp = 1/totalParticles;
    }
    for(int i =0; i<nFilters; i++){
        filters[i].nParticles = (totalParticles/nFilters);
        filters[i].weight = filters[i].weightPrev = 1.0/nFilters;
    }
}

/* for each detection, do the p(x_t | x_t-1) transation distribution, weights excluded*/
 void particleFilter::transition(int w, int h, gsl_rng* rng){ //  w and h are the max coodinate value for each axis
     for(int i=0; i<totalParticles; i++){
        particle pi = particles[i];
        particle pn;
        double x =  pi.x +  gsl_ran_gaussian(rng, 1);
        double y =  pi.y +  gsl_ran_gaussian(rng, 1);
        pn.x = (float) MAX( 0.0, MIN((float) w - width, x));
        pn.y = (float) MAX( 0.0, MIN((float) h - height, y));
        pn.xp = pi.x;
        pn.yp = pi.y;
        pn.id = pi.id;
        double tempx = pn.velocityX + gsl_ran_gaussian(rng, sigmaX);
        double tempy = pn.velocityY + gsl_ran_gaussian(rng, sigmaX);
        pn.velocityXp = pn.velocityX;
        pn.velocityYp = pn.velocityY;
        pn.velocityX = tempx;
        pn.velocityY = tempy;
        particles[i] = pn;
     }
 }

 /*update transated particle weight based on similarities with the previous detected hog discriptor*/
 void particleFilter::updateWeight(Mat& frame,  const HOGDescriptor& hog ){
     for(int i = 0; i < totalParticles; i++){
         if(particles[i].id!=removefLAG){
            double temp = particles[i].wp;
            particles[i].wp = particles[i].w;
            particles[i].w = computerSim(hog, frame, particles[i].x, particles[i].y, 0)
                    *dynamic_prob(&particles[i])/proposalP(hog, frame, &particles[i], 0.5, 0);
            particles[i].w =computerSim(hog, frame, particles[i].x, particles[i].y, 0);
         }
    }
    cout << "finish calculation" << endl;

    //normalized particle weight
    double eachFilterSum[nFilters];
    for(int i=0; i < totalParticles; i++){
        int id = particles[i].id;
        if(id != removefLAG)
            eachFilterSum[id] += particles[i].w;
    }

     for(int i=0; i < totalParticles; i++){
        if(eachFilterSum[particles[i].id]!=0 && particles[i].id!= removefLAG){
            particles[i].wNormalized = particles[i].w / eachFilterSum[particles[i].id];
        }
        else particles[i].wNormalized = 0;
    }
    cout << "normalize finish" << endl;

    //update each filter's weight
    double eachFilterW[nFilters];
    for(int i=0; i< totalParticles; i++){
        int id = particles[i].id;
        if(id != removefLAG)
            eachFilterW[id] += particles[i].w;
    }
    double sigmaweight =0; // simga pi n t-1 * wnt
    for(int j=0; j<nFilters; j++){
        sigmaweight += filters[j].weightPrev * eachFilterW[j];
    }
    for(int j=0; j<nFilters; j++){
        double temp = filters[j].weight;
        filters[j].weight = (double)filters[j].weightPrev * eachFilterW[j] / sigmaweight;
        cout << "filter " << j << "weight is" << filters[j].weight << endl;
        filters[j].weightPrev = temp;
    }
 }


void particleFilter::resample(gsl_rng* rng){
    particle * newParticles;
    newParticles = (particle*) malloc(totalParticles * sizeof(particle));
    int index =(int)gsl_rng_uniform (rng)* totalParticles;
    double beta = 0.0;
    double maxWeight =0;
    for (int i=0; i < totalParticles; i++) {
        if(particles[i].id != removefLAG)
            maxWeight = max(maxWeight, particles[i].wNormalized* filters[particles[i].id].weight);
    }
    for(int j=0; j < totalParticles; j++){
        if(particles[j].id == removefLAG){
            newParticles[j] = particles[j];
            continue;
        }
        beta += (rng, 0, 2*maxWeight);
        while(beta > particles[index].wNormalized* (filters[particles[index].id].weight)){
            beta -= particles[index].wNormalized * (filters[particles[index].id].weight);
            index = (index+1) % totalParticles;
        }
        while(particles[index].w==0){
                 index = (index+1) % (totalParticles);
        }
        newParticles[j] = particles[index];
    }
    cout << "Resample finished partly" << endl;
    for(int i=0; i<totalParticles; i++){
        particles[i] = newParticles[i];
        //particles[i].w = particles[i].wNormalized = 1/totalParticles;
    }
    free(newParticles);
    vector<int> countP1;
    for(int i=0; i<nFilters; i++){
        countP1.push_back(0);
    }
    for(int i=0; i< totalParticles; i++){
        if(particles[i].id!=removefLAG)
            countP1.at(particles[i].id)++;
    }
    for(int i=0; i< nFilters; i++){
        filters[i].nParticles = countP1.at(i);
    }
    for(int i=0; i< nFilters; i++){
        if(filters[i].nParticles > 20){
            cout << "call the remove function" << endl;
            removeParticles(i);
        }
    }
    vector<int> countP;
    for(int i=0; i<nFilters; i++){
        countP.push_back(0);
    }
    for(int i=0; i< totalParticles; i++){
        if(particles[i].id!=removefLAG)
            countP.at(particles[i].id)++;
    }
    for(int i=0; i< nFilters; i++){
        filters[i].nParticles = countP.at(i);
    }
    for(int i=0; i< nFilters;i++)
        cout << "n particle value is " << filters[i].nParticles << endl;

    int Remove = 0 ;
    for(int i=0; i< totalParticles; i++){
        if(particles[i].id==removefLAG)
            Remove ++ ;
    }
    for(int i=0; i< totalParticles; i++){
        if(particles[i].id==removefLAG){
            particles[i].w = particles[i].wNormalized = 1.0/(totalParticles - Remove);
        }
    }
}

void particleFilter::removeParticles(int i){
    map <double, int> m;
    for(int j=0; j<totalParticles; j++){
        if(particles[j].id == i){
            m[particles[j].w] = j;
        }
    }
    //sort the map by key
    int count = 0;
    typedef map<double, int>::iterator it_type;
    for(it_type iterator = m.begin(); iterator != m.end(); iterator++) {
        count ++;
        if(count <= filters[i].nParticles - 20){
           int id = iterator-> second;
           //cout << "check order or not " <<  iterator-> first << endl;
           particles[id].id = removefLAG;
           particles[id].x = particles[id].y = particles[id].w = particles[id].wNormalized = 0;
       }
    }
    filters[i].nParticles = 20;
}


//http://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#gsc.tab=0
//compute sim based on the exsiting hog descriptor and detector apperance model
double particleFilter::computerSim(const HOGDescriptor& hog, Mat& frame, int x, int y, double hitThreshold){
        vector<Point> searchLoc;
        vector<Point> foundLoc;
        vector<double> weights;
        searchLoc.push_back(Point((double)x, (double)y));
        hog.detect(frame, foundLoc, weights, hitThreshold, Size(24,48), Size(1,1), searchLoc);
        if(weights.empty())
            return 0;
        return MAX(0,weights.at(0)-2.8);
}

//dynamic model
double particleFilter::dynamic_prob(particle *p){
    double xProbability = gsl_ran_gaussian_pdf (p->x - p->xp, sigmaX);
    double yProbability = gsl_ran_gaussian_pdf (p->y - p->yp, sigmaY);
//return sqrt(xProbability*yProbability);
  return xProbability;
}

//proposal distribution
double particleFilter::proposalP(const HOGDescriptor& hog, Mat& frame, particle *p, double alpha, double hitThreshold){
    double apperance =  computerSim(hog, frame, p -> x , p -> y, hitThreshold);
    double dynamic = dynamic_prob(p);
return alpha * apperance + (1-alpha)*dynamic;
}
void particleFilter::displayParticles(Mat& frame){
    for(int i=0; i<totalParticles; i++){
        //if(particles[i].id != removefLAG){
            int x0 = cvRound(particles[i].x);
            int y0 = cvRound(particles[i].y);
            Rect r = Rect(min(720-width, x0), min(480-height,y0), width, height);
            rectangle(frame, r, Scalar(64, 255, 64), 1);
        //}
    }
    imshow("result",frame);
    waitKey(4000);
}


