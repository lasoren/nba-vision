#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctype.h>
#include <cmath>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"

using namespace std;
using namespace cv;

namespace nba_vision {

const char windowName[] = "Optical Flow";
const Size winSize(31,31);
const TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03)
class OpticalFlow{
public:
	OpticalFlow(bool debug=false);
	// Compute Optical flow with given points.
	// Compute Optical flow without points given (we calculate our own points).	
	void computeOpticalFlow(Mat current_frame);
private:
	bool debug_;
	void drawFlow(Point2f point_a, Point2f point_b);
	void buildPointGrid(Mat current_frame);
	double computeDistance(Point2f point_a, Point2f point_b);
	void computeAverageOpticalFlow(vector<double> distance);
	void computeSTDOpticalFlow(vector<double> distance);	
	// Maximum number of reference points
	Mat previous_frame;
	// Points used to track optical flowPoint2f 
	vector<Point2f> points[2];
	// metrics to determine if optical flow is due to camera motion
	double average_optical_flow;
	double std_optical_flow;
};

}
#endif
