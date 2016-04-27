#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <iostream>
typedef struct particle {
    double x; /* current x coordinate */
    double xp; /* previous x coordinate */
    double w; /*weight of each particle which could be used for resample, a sequential importance sampling*/
    double wp; /* previous weight which could be used for updating cur weight*/
} particle;

class SimPFilter{
public:
	int nParticles;
    gsl_rng* rng;
    double weight;
    int objectID;
	particle particles[100];
	
	SimPFilter();
    ~SimPFilter();
    void initParticles(double initX,  gsl_rng* rng);
    void transition(gsl_rng* rng);
    double measure_prob(particle *p,double obser_y, gsl_rng* rng);
    void resample(gsl_rng* rng);
    void updateWeight(double obser_y, gsl_rng* rng);
};
