#ifndef BSOPENCV_H
#define BSOPENCV_H
#endif // BSOPENCV_H
#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__
#endif
#include "opencv2/cvconfig.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <list>
#ifndef __OPENCV_BACKGROUND_SEGM_HPP__
#define __OPENCV_BACKGROUND_SEGM_HPP__
#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/video/video_tegra.hpp"
#endif
#endif
#undef K
#undef L
#undef T
#include <float.h>
#include <iostream>
#include <sstream>
#include <string>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cctype>
#include <string>
using namespace std;
using namespace cv;

class CV_EXPORTS_W MyBackgroundSubtractor : public cv::Algorithm
{
public:
    virtual ~MyBackgroundSubtractor();

    CV_WRAP_AS(apply) virtual void operator()(InputArray image, OutputArray fgmask, double learningRate=0);

    virtual void getBackgroundImage(OutputArray backgroundImage) const;
};

class CV_EXPORTS_W BackgroundSubtractorMOG : public MyBackgroundSubtractor
{
public:

    CV_WRAP BackgroundSubtractorMOG();

    CV_WRAP BackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio, double noiseSigma=0, int x=0, int y=0);

    virtual ~BackgroundSubtractorMOG();

    virtual void operator()(InputArray image, OutputArray fgmask, double learningRate=0, int x=0, int y=0);


    virtual void initialize(Size frameSize, int frameType);

    Mat processFrame( Mat frame);

    void processVideo(char* videoFilename);

protected:
    Size frameSize;
    int frameType;
    Mat bgmodel;
    int nframes;
    int history;
    int nmixtures;
    double varThreshold;
    double backgroundRatio;
    double noiseSigma;
};


