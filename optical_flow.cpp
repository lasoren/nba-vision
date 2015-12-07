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
	if( points[0].empty() ){
		buildPointGrid(points[0], current_frame);
	}
	if( !previous_frame.empty() ){
		vector<double> distance(points[0].size());
		vector<uchar> status;
		vector<double> err;
		calcOpticalFlowPyrLK(previous_frame, current_frame, points[0],
					points[1], status, err, winSize, 3, termcrit, 0, 0.01);
		for( int i = 0; i < points[1].size(); i++ ){
                	if( !status[i] )
                    		continue;
			distance[i](computeDistance(points[0][i], points[1][i]));
	    	}
		computeAverageOpticalFlow(distance);
		computeSTDOpticalFlow(distance);
		for( int i = 0; i < points[1].size(); i++ ){
                	if( !status[i])
                    		continue;
			double diff = abs(distance[i] - average_optical_flow);
			if (diff > average_optical_flow + std_optical_flow ||
				 diff < average_optical_flow - std_optical_flow ){
				drawFlow(points[0][i], points[1][i]);
			}
	    	}
		//debug	
		if ( debug_ ){
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

void OpticalFlow::buildPointGrid(Mat current_frame){
	int rows = current_frame.rows;
	int cols = current_frame.cols;
	cout << "rows,cols: " << rows << ", " << cols << endl;
	for (int x = 0; x < cols; x+=10){
		for (int y = 0; y < cols; y+=20){
			Point2f p(x, y);
			points[0].push_back(p);
		}
	}
}

double OpticalFlow::computeDistance(Point2f point_a, Point2f point_b){
	return sqrt(pow(point_b.x - point_a.x, 2.0) + pow(point_b.y - point_a.y, 2.0));	
}

void OpticalFlow::computeAverageOpticalFlow(vector<double> distance){
	double sum = 0.0;
	double count = 0.0;
	(for int i=0; i < distance.size(); i++){
		if (distance[i]){
			sum+=distance;
			count++;	
		}
	}
	average_optical_flow = sum/count;
		
}

void OpticalFlow::computeSTDOpticalFlow(vector<double> distance);
	//need to do
}


