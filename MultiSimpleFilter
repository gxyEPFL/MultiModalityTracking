#include "SimPFilter.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <vector>

using namespace std;
double sigmaX = 0.1;
double sigmaY = 0.1;

SimPFilter::SimPFilter() {
    totalParticles = 100;
    nFilters = 2;
    gsl_rng_env_setup();
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, time(NULL));
}

SimPFilter::~SimPFilter() {
    
}

void SimPFilter::initParticles(double initX, gsl_rng* rng){
    for (int i=0; i < totalParticles; i++) {
        particles[i].xp = particles[i].x = 0.5 + gsl_ran_gaussian(rng, sigmaX*sigmaX);  
        if(i > totalParticles/2)
            particles[i].xp = particles[i].x = -0.5 + gsl_ran_gaussian(rng, sigmaX*sigmaX); 
        particles[i].wNormalized = 1/(totalParticles/nFilters);
        int j = (int)i/(totalParticles/nFilters);
        particles[i].id = j;
        filters[j].nParticles = (totalParticles/nFilters);
        particles[i].w = particles[i].wp = 1/totalParticles;
        
    }
    for(int i =0; i<nFilters; i++){
        filters[i].weight = filters[i].weightPrev = 1.0/nFilters;
        
    }
}

void SimPFilter::transition(gsl_rng* rng){
	for (int i=0; i < totalParticles; i++) {
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
	for (int i=0; i < totalParticles; i++) {
        //particles[i].id = particles[i].id;
		particles[i].wp = particles[i].w;
		particles[i].w = measure_prob(&particles[i], obser_y, rng);
    }

     //normalized particle weight
    double eachFilterSum[nFilters];
    for(int i=0; i < totalParticles; i++){
        int id = particles[i].id;
        eachFilterSum[id] += particles[i].w;
    }

     for(int i=0; i < totalParticles; i++){
        particles[i].wNormalized = particles[i].w / eachFilterSum[particles[i].id];
    }

    //update each filter's weight
    double eachFilterW[nFilters];
    for(int i=0; i< totalParticles; i++){
        //cout << "prticle i's weight1 "<<particles[i].w << endl;
        int id = particles[i].id;
        eachFilterW[id] += particles[i].wNormalized;
    }
    double sigmaweight =0; // simga pi n t-1 * wnt
    for(int j=0; j<nFilters; j++){
        sigmaweight += filters[j].weightPrev * eachFilterW[j];
    }
    for(int j=0; j<nFilters; j++){
        double temp = filters[j].weight;
        filters[j].weight = filters[j].weightPrev * eachFilterW[j] / sigmaweight;
        filters[j].weightPrev = temp;
    }
}

void SimPFilter::resample(gsl_rng* rng){
	particle * newParticles;
	newParticles = (particle*) malloc(totalParticles * sizeof(particle));
	int index =(int)gsl_rng_uniform (rng)* totalParticles;
	double beta = 0.0;
	double maxWeight =0;
    for (int i=0; i < totalParticles; i++) {
    	maxWeight = max(maxWeight, particles[i].wNormalized* filters[particles[i].id].weight);
    }
    for(int j=0; j < totalParticles; j++){
        beta += (rng, 0, 2*maxWeight);
        while(beta > particles[index].wNormalized * (filters[particles[index].id].weight)){
            beta -= particles[index].wNormalized * (filters[particles[index].id].weight);
            index = (index+1) % totalParticles;
        }
        newParticles[j] = particles[index];
    }
    for(int i=0; i<totalParticles; i++){
        particles[i] = newParticles[i];
        particles[i].w = particles[i].wNormalized = 1/totalParticles;
    }
    //update each filter particles number
    vector<int> countP;
    for(int i=0; i<nFilters; i++){
        countP.push_back(0);
    }
    for(int i=0; i< totalParticles; i++){
        countP.at(particles[i].id)++;
    }
    for(int i=0; i< nFilters; i++){
        filters[i].nParticles = countP.at(i);
    }
    for(int i=0; i< nFilters;i++)
        cout << "n particle value is " << filters[i].nParticles << endl;
    free(newParticles);
}
