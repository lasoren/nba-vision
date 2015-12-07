#include "optical_flow.h"

using namespace cv;
using namespace std;

namespace nba_vision {


OpticalFlow::OpticalFlow(bool debug){
	debug_ = debug;
	if (debug_){
		namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	}
}

void OpticalFlow::computeOpticalFlow(Mat current_frame){
	cvtColor(current_frame, current_frame, COLOR_BGR2GRAY);
	if(points[0].empty()){
		buildPointGrid(points[0], current_frame);
	}
	if(!previous_frame.empty()){
		vector<uchar> status;
		vector<float> err;
		calcOpticalFlowPyrLK(previous_frame, current_frame, points[0],
					points[1], status, err, winSize, 3, termcrit, 0, 0.01);
		for( int i = 0; i < points[1].size(); i++ ){
                	if( !status[i] )
                    		continue;
			else
		    		drawFlow(points[0][i], points[1][i]);
	    	}
		//debug	
		if (debug_){
			imshow(windowName, previous_frame);
		}
	}
	// update for next frame
	//swap(points[1], points[0]); // new points become old points
	previous_frame = current_frame; // current frame becomes previous frame.
}

void OpticalFlow::drawFlow(Point2f point_a, Point2f point_b){
	Point p0( ceil( point_a.x ), ceil( point_a.y ) );
	Point p1( ceil( point_b.x ), ceil( point_b.y ) );
	line( previous_frame, p0, p1, CV_RGB(255,255,255), 2 );	
}

void OpticalFlow::buildPointGrid(vector<Point2f>& points, Mat current_frame){
	int rows = current_frame.rows;
	int cols = current_frame.cols; // column count seems to be incorrect not sure why
	cout << "rows,cols: " << rows << ", " << cols << endl;
	for (int x = 0; x < cols; x+=10){
		for (int y = 0; y < cols; y+=20){
			Point2f p(x, y);
			points.push_back(p);
			//cout << "point at P(" << i << ", " << j << ")" << endl;
		}
	}
}

}


