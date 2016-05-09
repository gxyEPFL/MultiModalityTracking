#include <iostream>
//#include "bsopencv.h"
#include "opencv2/cvconfig.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <float.h>
#include <iostream>
#include <sstream>
#include <string>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cctype>
#include <string>
#include "bsopencv.h"
#include "particleFilter.h"
#include "utils.h"
//#include "descriptor.h"
#include "adaboost.h"
using namespace std;

/*int main(int argc, char *argv[])
{
    return 0;
    BackgroundSubtractorMOG pMOG = BackgroundSubtractorMOG();//args[1] is the path for video file
    pMOG.processVideo(argv[1]);
    destroyAllWindows();
    return 0;

}*/

#define PI 3.14159265

int time_step = 100;
int nParticles = 200;
int nFilters = 10;
static string descriptorVectorFile = "/Users/xinyiguo/SonyCPPGithub/tracker/genfiles/descriptorvector.dat";
using namespace std;

int main(int argc, char *argv[]) {
    particleFilter* spf = new particleFilter;
    cout<<"hello world"<<endl;
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
        }
        if(countFrame==3){
            spf->initParticles(detected, spf->rng);
            spf->displayParticles(frame);
        }
        if(countFrame>3){
            cout <<"before transition" << endl;
            spf->transition(720, 480, spf->rng);
            cout << "before display"<< endl;
            //spf->displayParticles(frame);
            cout << "before update weight" << endl;
            spf->updateWeight(frame, hog);
            cout<<"update weiht succeed"<<endl;
            spf->resample(spf->rng);
            cout<<"resmaple succeed" << endl;
            spf->displayParticles(frame);
            cout<<"display succeed"<<endl;
        }
    }
}
