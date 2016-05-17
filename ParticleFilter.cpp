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
    totalParticles = 300;
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
    nFilters = orgTarget;
    int eachFilter = totalParticles/nFilters;
    for (int i=0; i < totalParticles; i++) {
        int j= i/eachFilter;
        particles[i].xp = particles[i].x = region.at(j % orgTarget).x + gsl_ran_gaussian(rng, sigmaX*sigmaX); //left top of the target
        particles[i].yp = particles[i].y = region.at(j % orgTarget).y + gsl_ran_gaussian(rng, sigmaY*sigmaY); //left top of the target
        particles[i].velocityX = particles[i].velocityXp =   gsl_ran_gaussian(rng, sigmaX*sigmaX); // maybe wrong
        particles[i].velocityY = particles[i].velocityYp =   gsl_ran_gaussian(rng, sigmaY*sigmaY);
        particles[i].wNormalized = (double)1/(totalParticles/orgTarget);
        particles[i].id = j;
        particles[i].w = particles[i].wp = 1.0;
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
        double x =  pi.x +  3 * gsl_ran_gaussian(rng, sigmaX*sigmaX);
        double y =  pi.y +  2 * gsl_ran_gaussian(rng, sigmaY*sigmaY);
        pn.x = (float) MAX( 0.0, MIN((float) w - width, x));
        pn.y = (float) MAX( 0.0, MIN((float) h - height, y));
        pn.xp = pi.x;
        pn.yp = pi.y;
        pn.id = pi.id;
        pn.w = pi.w;
        pn.wNormalized = pi.wNormalized;
        pn.wp = pi.wp;
        double tempx = pn.velocityX + 3 * gsl_ran_gaussian(rng, sigmaX*sigmaX);
        double tempy = pn.velocityY + 2 * gsl_ran_gaussian(rng, sigmaY*sigmaY);
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
            particles[i].wp = particles[i].wNormalized;
            particles[i].w = temp * computeSim(hog, frame, particles[i].x, particles[i].y, 0)
                     * dynamic_prob(&particles[i])
                     /proposalP(hog, frame, &particles[i], 0.99, 0);
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
    cout << "normalize finish 1 " << endl;
     for(int i=0; i < totalParticles; i++){
        if(eachFilterSum[particles[i].id]!=0 && particles[i].id!= removefLAG){
            particles[i].wNormalized = particles[i].w / eachFilterSum[particles[i].id];
        }
        else particles[i].wNormalized = 0;
    }

    //update each filter's weight
    double eachFilterW[nFilters];
    for(int i=0; i< totalParticles; i++){
        int id = particles[i].id;
        if(id != removefLAG)
            eachFilterW[id] += particles[i].w; //msitake
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

/*resample the particles based on the weight, use wheel round method */
void particleFilter::resample(gsl_rng* rng){
    particle * newParticles;
    newParticles = (particle*) malloc(totalParticles * sizeof(particle));
    double beta = 0.0;
    double maxWeight =0;
    int newCount = 0;
    for(int i=0; i<nFilters; i++){
                int indexArray[filters[i].nParticles];
                int count = 0;
                for(int j =0; j<totalParticles; j++){
                    if(particles[j].id == i){
                        indexArray[count] = j;
                        count ++;
                    }
                }
                double beta = 0.0;
                double maxWeight =0;
                for (int i=0; i < count; i++) {
                    maxWeight = max(maxWeight,  particles[indexArray[i]].wNormalized);
                }
                int index =(int)gsl_rng_uniform (rng)*count;
                for(int k=0 ; k < totalParticles * filters[i].weight; k++){
                        beta += gsl_rng_uniform (rng) * 2 * maxWeight;
                        while(beta > particles[indexArray[index]].wNormalized){
                            beta -= particles[indexArray[index]].wNormalized;
                            index = (index+1) %count;
                        }
                        newParticles[k + newCount] = particles[indexArray[index]];
                        //cout << "choose " << indexArray[index] << " particle" << endl;
                }
                    newCount += totalParticles * filters[i].weight;
            }
            for(int i=0; i<totalParticles; i++){
                particles[i] = newParticles[i];
            }
            free(newParticles);
    cout << "Resample finished partly" << endl;
    vector<int> countP1;
    for(int i=0; i<nFilters; i++){
        countP1.push_back(0);
    }
    for(int i=0; i< totalParticles; i++){
        if(particles[i].id!=removefLAG)
            countP1.at(particles[i].id) ++;
    }
    for(int i=0; i< nFilters; i++){
        filters[i].nParticles = countP1.at(i);
    }
    for(int i=0; i< nFilters; i++){
        if(filters[i].nParticles > 50){
            cout << "call the remove function" << endl;
            //removeParticles(i);
        }
    }
    vector<int> countP;
    for(int i=0; i<nFilters; i++){
        countP.push_back(0);
    }
    for(int i=0; i< totalParticles; i++){
        if(particles[i].id!=removefLAG)
            countP.at(particles[i].id) = countP.at(particles[i].id)+1;
    }
    for(int i=0; i< nFilters; i++){
        filters[i].nParticles = countP.at(i);
    }
    for(int i=0; i< nFilters;i++)
        cout << "n particle value is " << filters[i].nParticles << endl;

    int Remove = 0 ;
    for(int i=0; i< totalParticles; i++){
        if(particles[i].id==removefLAG){
            Remove ++ ;
        }
    }
    for(int i=0; i< totalParticles; i++){
        if(particles[i].id==removefLAG){
            particles[i].wp = particles[i].w =particles[i].wNormalized =0;
        }
    }
    for(int i=0; i<totalParticles; i++){
               particles[i].wp = particles[i].w = particles[i].wNormalized = 1.0/filters[particles[i].id].nParticles;
           }
}

void particleFilter::removeParticles(int i){
    map <double, int> m;
    int countTotal = 0;
    for(int j=0; j<totalParticles; j++){
        if(particles[j].id == i){
            m[particles[j].w] = j;
            countTotal++;
        }
    }
    cout << "we need to remove filter" << i << "with particles " << countTotal;
    //sort the map by key
    int count = 0;
    typedef map<double, int>::iterator it_type;
    for(it_type iterator = m.begin(); iterator != m.end(); iterator++) {
        count ++;
        if(count <= countTotal - 50){
           int id = iterator-> second;
           particles[id].id = removefLAG;
           particles[id].x = particles[id].y = particles[id].w = particles[id].wNormalized = 0;
       }
    }
    filters[i].nParticles = 50;
}


//http://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#gsc.tab=0
/*compute sim based on the exsiting hog descriptor and detector apperance model*/
double particleFilter::computeSim(const HOGDescriptor& hog, Mat& frame, int x, int y, double hitThreshold){
        vector<Point> searchLoc;
        vector<Point> foundLoc;
        vector<double> weights;
        searchLoc.push_back(Point((double)x, (double)y));
        hog.detect(frame, foundLoc, weights, hitThreshold, Size(24,48), Size(1,1), searchLoc);
        if(weights.empty())
            return 0;
        return MIN(1,(MAX(0,weights.at(0)-2)));
}

/*dynamic model*/
double particleFilter::dynamic_prob(particle *p){
    double xProbability = gsl_ran_gaussian_pdf ((p->x - p->xp)/3.0, sigmaX);
    double yProbability = gsl_ran_gaussian_pdf ((p->y - p->yp)/2.0, sigmaY);
 // return xProbability;
return sqrt(xProbability*yProbability);
}

/*proposal distribution*/
double particleFilter::proposalP(const HOGDescriptor& hog, Mat& frame, particle *p, double alpha, double hitThreshold){
    double apperance =  computeSim(hog, frame, p -> x , p -> y, hitThreshold);
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
    waitKey(100);
}

/*void particleFilter::addParticles(const HOGDescriptor& hog, Mat& bsRes, Mat& frame){
    vector <Rect> detected = getDetected(hog, 2.8, bsRes, frame);
    set<int> removedID;
    for(int i=0; i<particles.size(); i++){
        if(particles[i].id == removefLAG){
            removedID.insert(i);
        }
    }

}

bool particleFilter::checkNeighbor(Rect adaP, Rect trackP, double threshold){
    double dis = sqrt((adaP.x - trackP.x)^2 +( adaP.y - trackP.y)^2);
        if(dis<threshold) return true;
    return false;
}

vector<Rect> particleFilter::retrieveRects(){

}*/


