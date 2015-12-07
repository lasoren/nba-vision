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
		buildPointGrid(current_frame);
	}
	if( !previous_frame.empty() ){
		vector<double> distance(points[0].size());
		vector<uchar> status;
		vector<float> err;

		calcOpticalFlowPyrLK(previous_frame, current_frame, points[0],
					points[1], status, err, winSize, 3, termcrit, 0, 0.01);
		
		for( int i = 0; i < points[1].size(); i++ ){
                	if( !status[i] )
                    		continue;
			distance[i] = (computeDistance(points[0][i], points[1][i]));
	    	}
		computeAverageOpticalFlow(distance);
		computeSTDOpticalFlow(distance);

		cout << "average: " << average_optical_flow << " std: " << std_optical_flow
		<< endl;

		for( int i = 0; i < points[1].size(); i++ ){
                	if( !status[i])
                    		continue;
			double diff = abs(distance[i] - average_optical_flow);
			if (diff > (std_optical_flow*1.3)){
				drawFlow(points[0][i], points[1][i], false);
			}
			else{
				drawFlow(points[0][i], points[1][i], true);
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

void OpticalFlow::drawFlow(Point2f point_a, Point2f point_b, bool camera_motion){
	Point p0( ceil( point_a.x ), ceil( point_a.y ) );
	Point p1( ceil( point_b.x ), ceil( point_b.y ) );
	if (camera_motion){
		line( previous_frame, p0, p1, CV_RGB(0,0,0), 2 );
	}
	else{	
		line( previous_frame, p0, p1, CV_RGB(255,255,255), 2 );
	}
}

void OpticalFlow::buildPointGrid(Mat current_frame){
	int rows = current_frame.rows;
	int cols = current_frame.cols;
	for (int x = 0; x < cols; x+=10){
		for (int y = 0; y < cols; y+=10){
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
	for (int i=0; i < distance.size(); i++){
		if (distance[i]){
			sum+=distance[i];
			count++;	
		}
	}
	average_optical_flow = sum/count;
		
}

void OpticalFlow::computeSTDOpticalFlow(vector<double> distance){
	double sum = 0.0;
	double count = 0.0;
	for(int i=0; i < distance.size(); i++){
		if (distance[i]){
			sum+=pow(distance[i] - average_optical_flow, 2.0);
			count++;	
		}
	}
	std_optical_flow = sqrt(sum/count);
}

}



