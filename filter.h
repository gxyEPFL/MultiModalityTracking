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
    double xAda;
    double y; /* current y coordinate */
    double yAda;
    double xp; /* previous x coordinate */
    double yp; /* previous y coordinate */
    double velocityX;
    double velocityY;
    double velocityXp;
    double velocityYp;
    double w; /*weight of each particle which could be used for resample, a sequential importance sampling*/
    double wp; /* previous weight which could be used for updating cur weight*/
    double wNormalized;
    void transition(int w, int h, gsl_rng* rng);
} particle;

class filter{
public:
    double sigmaX = 1;
    double sigmaY = 1;
    int height = 48;
    int width = 24;
    int w = 720;
    int h = 480;
    gsl_rng* rng;
    double  pi =0;
    double  piPrev = 0;
    int objectID;
    bool available;
    int capacity=30; // number of particles inside one group(30)
    particle particles[30];

    filter();
    ~filter();
    void initParticles(Rect region, double prob);
    void transition(double alpha);
    void updateWeight(Mat& frame, const HOGDescriptor& hog, double alpha);
    void resample();
    void displayParticles(Mat& frame, Scalar s);
    Rect getRecFilter();
    static double computeSim(const HOGDescriptor& hog, Mat& frame, int x, int y, double hitThreshold);

private:
    double proposalP(const HOGDescriptor& hog, Mat& frame, particle *p, double alpha, double hitThreshold);
    double dynamic_prob(particle *p);
};
