#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__
#endif
#include "bsOpenCV.h"
#include "opencv2/cvconfig.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <list>
#ifndef __OPENCV_BACKGROUND_SEGM_HPP__
#define __OPENCV_BACKGROUND_SEGM_HPP__
#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/video/video_tegra.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
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
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cctype>
#include <string>


using namespace std;
using namespace cv;

static const int defaultNMixtures = 5;
static const int defaultHistory = 200;
static const double defaultBackgroundRatio = 0.8;
static const double defaultVarThreshold = 2.5;
static const double defaultNoiseSigma = 30*0.5;
static const double defaultInitialWeight = 0.05;
static double learning = 0.0001;
const static Scalar colors[] =  { CV_RGB(0,0,255),CV_RGB(0,255,0),CV_RGB(255,0,0),CV_RGB(255,255,0),CV_RGB(255,0,255),CV_RGB(0,255,255),CV_RGB(255,255,255),CV_RGB(128,0,0),CV_RGB(0,128,0),CV_RGB(0,0,128),CV_RGB(128,128,128),CV_RGB(0,0,0)};
void MyBackgroundSubtractor::operator()(InputArray, OutputArray, double)
{
}

void MyBackgroundSubtractor::getBackgroundImage(OutputArray) const
{
}

MyBackgroundSubtractor::~MyBackgroundSubtractor() {}
BackgroundSubtractorMOG::BackgroundSubtractorMOG(){
    frameSize = Size(0,0);
    frameType = 0;
    nframes = 0;
    nmixtures = defaultNMixtures;
    history = defaultHistory;
    varThreshold = defaultVarThreshold;
    backgroundRatio = defaultBackgroundRatio;
    noiseSigma = defaultNoiseSigma;
}

BackgroundSubtractorMOG::BackgroundSubtractorMOG(int _history, int _nmixtures,
                                                 double _backgroundRatio,
                                                 double _noiseSigma, int _x, int _y){
    frameSize = Size(0,0);
    frameType = 0;
    nframes = 0;
    nmixtures = min(_nmixtures > 0 ? _nmixtures : defaultNMixtures, 8);
    history = _history > 0 ? _history : defaultHistory;
    varThreshold = defaultVarThreshold;
    backgroundRatio = min(_backgroundRatio > 0 ? _backgroundRatio : 0.8, 1.);
    noiseSigma = _noiseSigma <= 0 ? defaultNoiseSigma : _noiseSigma;
}

BackgroundSubtractorMOG::~BackgroundSubtractorMOG(){
}

void BackgroundSubtractorMOG::initialize(Size _frameSize, int _frameType){
    frameSize = _frameSize;
    frameType = _frameType;
    nframes = 0;

    int nchannels = CV_MAT_CN(frameType);
    CV_Assert( CV_MAT_DEPTH(frameType) == CV_8U );
    bgmodel.create( 1, frameSize.height*frameSize.width*nmixtures*(2 + 2*nchannels), CV_32F );
    bgmodel = Scalar::all(0);
}


template<typename VT> struct MixData
{
    float sortKey;
    float weight;
    VT mean;
    VT var;
};


static void process8uC3( const Mat& image, Mat& fgmask, double learningRate,
                     Mat& bgmodel, int nmixtures, double backgroundRatio,
                     double varThreshold, double noiseSigma, int pixelX, int pixelY){
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)backgroundRatio, vT = (float)varThreshold;
    int K = nmixtures;
    const float w0 = (float)defaultInitialWeight;
    const float sk0 = (float)(w0/(defaultNoiseSigma*2*sqrt(3.)));
    const float var0 = (float)(defaultNoiseSigma*defaultNoiseSigma*4);
    const float minVar = (float)(noiseSigma*noiseSigma);
    MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;
    for( y = 0; y < rows; y++ ){
    const uchar* src = image.ptr<uchar>(y);
    uchar* dst = fgmask.ptr<uchar>(y);
    if( alpha > 0 ){
        for( x = 0; x < cols; x++, mptr += K ){
            float wsum = 0;
            Vec3f pix(src[x*3], src[x*3+1], src[x*3+2]);
            int kHit = -1, kForeground = -1;
            for( k = 0; k < K; k++ ){
                float w = mptr[k].weight;
                wsum += w;
                if( w < FLT_EPSILON )
                    break;
                Vec3f mu = mptr[k].mean;
                Vec3f var = mptr[k].var;
                Vec3f diff = pix - mu;
                float d2 = diff.dot(diff);
                if( d2 < vT*(var[0] + var[1] + var[2])){
                    wsum -= w;
                    float dw = alpha*(1.f - w);
                    mptr[k].weight = w + dw;
                    mptr[k].mean = mu + alpha*diff;
                    var = Vec3f(max(var[0] + alpha*(diff[0]*diff[0] - var[0]), minVar),
                                max(var[1] + alpha*(diff[1]*diff[1] - var[1]), minVar),
                                max(var[2] + alpha*(diff[2]*diff[2] - var[2]), minVar));
                    mptr[k].var = var;
                    mptr[k].sortKey = w/sqrt(var[0] + var[1] + var[2]);
                    for( k1 = k-1; k1 >= 0; k1-- ){
                        if( mptr[k1].sortKey >= mptr[k1+1].sortKey )
                            break;
                        std::swap( mptr[k1], mptr[k1+1] );
                    }
                    kHit = k1+1; // KHit record which Gaussian the cur pixel match.
                    break;
                }
            }
            if( kHit < 0 ){ // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
                if(x == pixelX && y==pixelY)
                kHit = k = min(k, K-1);
                wsum += w0 - mptr[k].weight;
                mptr[k].weight = w0;
                mptr[k].mean = pix;
                mptr[k].var = Vec3f(var0, var0, var0);
                mptr[k].sortKey = sk0;
            }
            else
                for( ; k < K; k++ )
                    wsum += mptr[k].weight;
            float wscale = 1.f/wsum;
            wsum = 0;
            for( k = 0; k < K; k++ ){
                mptr[k].weight *= wscale;
                wsum += mptr[k].weight;
                mptr[k].sortKey *= wscale;
                if( wsum > T && kForeground < 0 )
                    kForeground = k+1; // from this now on, the Gaussians are Foreground.
            }
            dst[x] = (uchar)(-(kHit >= kForeground));  //if kHit > kForground, then dst[x] =(uchar) -1 = 255, which is white,
            // not absorbed to the background. Otherwise, the new coming object finally integrated to the background.
            if(x == pixelX && y == pixelY){
                if(kHit < kForeground){
                    break;
                }
            }
        }
       }
    }
}

void BackgroundSubtractorMOG::operator()(InputArray _image, OutputArray _fgmask, double learningRate, int x, int y){
    Mat image = _image.getMat();
    bool needToInitialize = nframes == 0 || learningRate >= 1 || image.size() != frameSize || image.type() != frameType;
    if( needToInitialize )
        initialize(image.size(), image.type());

    CV_Assert( image.depth() == CV_8U );
    _fgmask.create( image.size(), CV_8U );
    Mat fgmask = _fgmask.getMat();
    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./min( nframes, history );
    CV_Assert(learningRate >= 0);
    if( image.type() == CV_8UC3 )
        process8uC3( image, fgmask, learningRate, bgmodel, nmixtures, backgroundRatio, varThreshold, noiseSigma, x, y);
    else
        CV_Error( CV_StsUnsupportedFormat, "Only 1- and 3-channel 8-bit images are supported in BackgroundSubtractorMOG" );
}

Mat BackgroundSubtractorMOG::processFrame(Mat frame){
    Mat destImage, fgMaskMOG;
    operator()(frame, fgMaskMOG, learning, 200, 300); // place to change
    rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
        cv::Scalar(255,255,255), -1);
    int dilation_type = MORPH_RECT;
    int dilation_size = 1;
    Mat dilationMask;
    Mat element = getStructuringElement( dilation_type,
                                   Size( 3 * dilation_size + 1, 3 * dilation_size + 1),
                                   Point( dilation_size, dilation_size ) );
    dilate(fgMaskMOG, dilationMask, element);
    Mat blurMask;
    for ( int i = 1; i < 11; i = i + 2 ){
        medianBlur(dilationMask, blurMask, i );
    }
    frame.copyTo(destImage, fgMaskMOG);
    for(int i = 0; i< destImage.cols; i++){
        for(int j=0; j<destImage.rows; j++){
            Vec3b color = destImage.at<Vec3b>(Point(i,j));
            if(color[0]==0 && color[1]==0 && color[2]==0){
                color[0]=255;
                destImage.at<Vec3b>(Point(i,j)) = color;
            }
        }
    }
    Mat blurDestImage;
    frame.copyTo(blurDestImage, blurMask);
    for(int i = 0; i< blurDestImage.cols; i++){
        for(int j=0; j< blurDestImage.rows; j++){
            Vec3b color = blurDestImage.at<Vec3b>(Point(i,j));
            if(color[0]==0 && color[1]==0 && color[2]==0){
                color[0]=255;
                blurDestImage.at<Vec3b>(Point(i,j)) = color;
            }
        }
    }
    imshow("destimage", blurDestImage);
    return blurDestImage;
}

void  BackgroundSubtractorMOG::processVideo(char* videoFilename) {
    VideoCapture capture(videoFilename);
    Mat frame;
    int keyboard;
    if(!capture.isOpened()){
        cerr << "Unable to open video file: " << videoFilename << endl;
        exit(EXIT_FAILURE);
    }
    while((char)keyboard != 'q' && (char)keyboard != 27){
        if(!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }
        Mat bsRes = processFrame(frame);
        keyboard = waitKey( 1 );
    }
    capture.release();
}

/*int main(int argc, char* argv[]) {
    BackgroundSubtractorMOG pMOG = BackgroundSubtractorMOG();
    //args[1] is the path for video file
    processVideo(argv[1], pMOG);
    destroyAllWindows();
    return 0;
}
*/
