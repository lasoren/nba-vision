#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctype.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"

using namespace std;
using namespace cv;

namespace nba_vision {

const char windowName[] = "Optical Flow";
const int MAX_COUNT=1000;
const Size subPixWinSize(10,10), winSize(31,31);
const TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);

class OpticalFlow{
public:
	OpticalFlow(bool debug=false);
	// Compute Optical flow with given points.
	// Compute Optical flow without points given (we calculate our own points).	
	void computeOpticalFlow(Mat current_frame);
private:
	bool debug_;
	void drawFlow(Point2f point_a, Point2f point_b);
	// Maximum number of reference points
	Mat previous_frame;
	// Points used to track optical flow
	vector<Point2f> points[2];
};

}
#endif
