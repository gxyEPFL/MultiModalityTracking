#include "SimPFilter.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <iostream>

using namespace std;
double sigmaX = 0.1;
double sigmaY = 0.1;

SimPFilter::SimPFilter() {
    nParticles = 100;
    gsl_rng_env_setup();
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, time(NULL));
}

SimPFilter::~SimPFilter() {
    
}

void SimPFilter::initParticles(double initX, gsl_rng* rng){
	for (int i=0; i < nParticles; i++) {
        if(i<nParticles/2)
            particles[i].xp = particles[i].x = 0.5 + gsl_ran_gaussian(rng, sigmaX*sigmaX);  //the centre of each region
        else 
            particles[i].xp = particles[i].x = -0.5 + gsl_ran_gaussian(rng, sigmaX*sigmaX);  //the centre of each region
        particles[i].w = 1/nParticles;
    }
}

void SimPFilter::transition(gsl_rng* rng){
	for (int i=0; i < nParticles; i++) {
        double temp = particles[i].x;
        particles[i].xp = temp; 
		particles[i].x  += gsl_ran_gaussian(rng, sigmaX*sigmaX);  //the centre of each region      
    }
}

double SimPFilter::measure_prob(particle *p, double obser_y, gsl_rng* rng){
	double x = p->x;
	return gsl_ran_gaussian_pdf (obser_y - x*x, sigmaY);
}

void SimPFilter::updateWeight(double obser_y, gsl_rng* rng){
	for (int i=0; i < nParticles; i++) {
		particles[i].wp = particles[i].w;
		particles[i].w = measure_prob(&particles[i], obser_y, rng);
	}
	double sum = 0;
    
    for (int i=0; i < nParticles; i++) {
        sum += particles[i].w;
    }
    
    for (int i=0; i < nParticles; i++) {
        particles[i].w /= sum;
    }
}

void SimPFilter::resample(gsl_rng* rng){
	particle * newParticles;
	newParticles = (particle*) malloc(nParticles * sizeof(particle));
	int index =(int)gsl_rng_uniform (rng)*nParticles;
	double beta = 0.0;
	double maxWeight =0;
    for (int i=0; i < nParticles; i++) {
    	maxWeight = max(maxWeight, particles[i].w);
    }
    for(int j=0; j < nParticles; j++){
        beta += (rng, 0, 2*maxWeight);
        while(beta > particles[index].w){
            beta -= particles[index].w;
            index = (index+1)%nParticles;
        }
        newParticles[j] = particles[index];
    }
    for(int i = 0; i<nParticles; i++){
        particles[i] = newParticles[i];
        particles[i].w = 1/nParticles;
    }
    free(newParticles);
}
