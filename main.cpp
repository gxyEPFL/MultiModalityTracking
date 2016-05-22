#include <string>
#include <float.h>
#include <iostream>
#include <sstream>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cctype>
#include "opencv2/cvconfig.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "bsopencv.h"
#include "particleFilter.h"
#include "utils.h"
//#include "descriptor.h"
#include "adaboost.h"
static string descriptorVectorFile = "/Users/xinyiguo/SonyCPPGithub/tracker/genfiles/descriptorvector.dat";

int main(int argc, char *argv[]) {
    particleFilter* spf = new particleFilter;
    vector<float> descriptorVector = getDescriptorVectorFromFile(descriptorVectorFile);
    BackgroundSubtractorMOG pMOG = BackgroundSubtractorMOG();
    HOGDescriptor hog;
    hog.winSize = Size(24, 48);
    std::cout << hog.blockSize << endl;
    hog.blockStride = Size(1,2);
    hog.setSVMDetector(descriptorVector); // set the hog descriptor
    VideoCapture capture("/Users/xinyiguo/Desktop/video/mv2_001.avi");
    Mat frame;
    int keyboard=0;
    if(!capture.isOpened()){
        cerr << "Unable to open video file: "  << endl;
        exit(EXIT_FAILURE);
    }
    int countFrame = 0;
    while((char)keyboard != 'q' && (char)keyboard != 27){
        countFrame ++;
        if(!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }
        Mat bsRes = pMOG.processFrame(frame);
        vector<Rect> detected;

        if(countFrame<=3){
            detected = getDetected(hog, 2.8, bsRes, frame);
            for(auto p: detected){
                cout<< p.x << " "<< p.y << endl;
            }
        }
        if(countFrame>=3){
            detected = getDetected(hog, 2.8, bsRes, frame);
            spf->process(frame,detected, hog);
        }
    }
}
