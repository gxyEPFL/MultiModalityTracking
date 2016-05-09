#ifndef PARTICLE_H
#define PARTICLE_H
#endif // PARTICLE_H
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
using namespace std;
using namespace cv;
typedef struct particle {
    double x; /* current x coordinate */
    double y; /* current y coordinate */
    double xp; /* previous x coordinate */
    double yp; /* previous y coordinate */
    double velocityX;
    double velocityY;
    double velocityXp;
    double velocityYp;
    //int width; /* width of region described by particle */
    //int height; /*height of region described by particle */
    double w; /*weight of each particle which could be used for resample, a sequential importance sampling*/
    double wp; /* previous weight which could be used for updating cur weight*/
    double wNormalized;
    int id;
} particle;

typedef struct filter{
    int nParticles;
    gsl_rng* rng;
    double weight;
    double weightPrev;
    int objectID;
} filter;

class particleFilter {
public:
    int totalParticles =200;
    int nFilters;
    gsl_rng* rng;
    filter filters[10];
    particle particles[200];
    particleFilter();
    ~particleFilter();
    void initParticles(vector<Rect> region,  gsl_rng* rng);
    void transition(int w, int h, gsl_rng* rng);
    void displayParticles(Mat* img, CvScalar nColor, CvScalar hColor, int param);
    void updateWeight(Mat& frame, const HOGDescriptor& hog);
    void resample(gsl_rng* rng);
    void displayParticles(Mat& frame);
    //vector<float> getDescriptorVectorFromFile(string fileName);
  private:
    double computerSim(const HOGDescriptor& hog, Mat& frame, int x, int y, double hitThreshold);
};
