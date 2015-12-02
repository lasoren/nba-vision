#include "optical_flow.h"

using namespace cv;
using namespace std;

namespace optical_flow {


OpticalFlow::OpticalFlow(bool debug){
	debug_ = debug;
	if (debug_){
		namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	}
}

OpticalFlow::computeOpticalFlow(Mat current_frame, vector<Point2f> current_points){
	points[1] = current_points;
	// checks if first frame
	if(!points[0].empty()){
		vector<uchar> status;
		vector<float> err;
		calcOpticalFlowPyrLK(previous_frame, current_frame, points[0],
					points[1], status, err, winSize, 3, termcrit, 0, 0.001);
		for( int i = 0; i < points[1].size(); i++ ){
                	if( !status[i] )
                    		continue;
			else
		    		drawFlow(points[0][i], points[1][i]);
	    	}
	// update for next frame
	if (debug_){
		imshow(windowName, previous_frame);
	}
	swap(points[1], points[0]); // new points become old points
	previous_frame = current_frame; // current frame becomes previous frame.
}

OpticalFlow::drawFlow(Point2f point_a, Point2f point_b){
	Point p0( ceil( point_a.x ), ceil( point_a.y ) );
	Point p1( ceil( point_b.x ), ceil( point_b.y ) );
	line( previous_frame, p0, p1, CV_RGB(255,255,255), 2 );

}

}


