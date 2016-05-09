#include "SimPFilter.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <vector>
#include "stdio.h"     
#include "math.h"
#include <stdlib.h>


#define PI 3.14159265
#define NUM_COMMANDS 3
using namespace std;

int time_step = 100;
int nParticles = 100;
vector<double> trueX;
vector<double> falseX;
vector<double> observationY;
/*double sigmaX = 0.1;
double sigmaY = 0.1;*/

void generate_ground_truth(){
	double sigmaY = 0.1;
	const gsl_rng_type * T;
  	gsl_rng * r;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	for(int i = 0; i< time_step; i++){
		double x = 0.5*(sin(2*i*PI/180)+1);
		trueX.push_back(x);
		falseX.push_back(-x);
    	//cout <<  gsl_ran_gaussian(r, sigmaY) << endl;
		double ober = x * x + 1 * gsl_ran_gaussian(r, sigmaY); 
		observationY.push_back(ober);
	}
}

void display(vector<double> trueX,vector<double> falseX, vector<double> observationY, vector<vector<double> > particleVec)
{
    char * commandsForGnuplot[] = {"set title \"TITLEEEEE\"", "plot 'data.temp'", "set yrange [-1:1]"};
    FILE * temp = fopen("data.temp", "w");
    FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");
    fprintf(gnuplotPipe, "plot '-' \n");
	int i;

	for (int i = 0; i < trueX.size(); i++)
	{
	  fprintf(gnuplotPipe, "%lf %lf\n", (double)i, observationY.at(i));
	  fprintf(gnuplotPipe, "%lf %lf\n", (double)i, trueX.at(i));
	  fprintf(gnuplotPipe, "%lf %lf\n", (double)i, falseX.at(i));
	  vector<double> particlePos =particleVec[i];
	  for(int j=0; j<particlePos.size(); j++)
	  	 fprintf(gnuplotPipe, "%lf %lf\n", (double)i, particlePos.at(j));
	}
	fprintf(gnuplotPipe, "e");
}

int main() {
	generate_ground_truth();
	SimPFilter* spf = new SimPFilter;
	vector<vector<double> > particleVec;
	for(int k=0; k <time_step; k++){
		if(k==0)
			spf->initParticles(0.0, spf->rng);
		vector<double> curParticle ;
        particleVec.push_back(curParticle);
		spf->transition(spf->rng);
		spf->updateWeight(observationY.at(k), spf->rng);
		spf->resample(spf->rng);
		if(k%5==0){
			for(int j=0; j<nParticles; j++){
			particleVec[k].push_back(spf->particles[j].x);
			}
		}
	}
	display(trueX, falseX, observationY, particleVec);
}


