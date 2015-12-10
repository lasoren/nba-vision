#include "optical_flow.h"

using namespace cv;
using namespace std;

namespace nba_vision {


Bucket::Bucket(double distance_mn, double distance_mx, double angle_mn, double angle_mx){
	distance_min = distance_mn;
	distance_max = distance_mx;
	angle_min = angle_mn;
	angle_max = angle_mx;
	count = 0;
}

Bucket::Bucket(){
	distance_min = NULL;
	distance_max = NULL;
	angle_min = NULL;
	angle_max = NULL;
	count = 0;
}

bool Bucket::inBucket(double distance, double angle){
	if (distance < distance_max && distance >= distance_min &&
		 angle < angle_max && angle >= angle_min){
		return true;
	}
	return false;
}

int Bucket::getCount(){
	return count;
}

void Bucket::resetCount(){
	count = 0;
}

void Bucket::incrementCount(){
	count++;
}

OpticalFlow::OpticalFlow(bool debug){
	debug_ = debug;
	if (debug_){
		namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	}
}

void OpticalFlow::computeOpticalFlow(Mat& cf){
	Mat current_frame;
	cvtColor(cf, current_frame, COLOR_BGR2GRAY);
	if( points[0].empty() ){
		buildPointGrid(current_frame);
	}
	if (buckets.empty()){
		buildBuckets(6, 30.0, 10);
		cout <<"first bucket angle :" << buckets[0].angle_max << endl;
	}
	if( !previous_frame.empty() ){
		vector<double> distance(points[0].size()), angle(points[0].size());
		vector<uchar> status;
		vector<float> err;

		calcOpticalFlowPyrLK(previous_frame, current_frame, points[0],
					points[1], status, err, winSize, 3, termcrit, 0, 0.01);	
		for( int i = 0; i < points[1].size(); i++ ){
                	if( !status[i] )
                    		continue;
			distance[i] = computeDistance(points[0][i], points[1][i]);
			angle[i] = computeAngle(points[0][i], points[1][i]);
			status[i] = assignBucket(distance[i], angle[i]);
	    	}
		Bucket max_bucket = maxBucket();
		cout << max_bucket.getCount() << endl;
		for( int i = 0; i < points[1].size(); i++ ){
                	if( !status[i])
                    		continue;
			if(!max_bucket.inBucket(distance[i], angle[i])){
				drawFlow(points[0][i], points[1][i], false, cf);
			}
	    	}
		//debug	
		if ( debug_ ){
			imshow(windowName, previous_frame);
		}
	}
	// update for next frame
	previous_frame = current_frame; // current frame becomes previous frame.
}

void OpticalFlow::drawFlow(Point2f point_a, Point2f point_b, bool camera_motion, Mat& cf){
	Point p0( ceil( point_a.x ), ceil( point_a.y ) );
	Point p1( ceil( point_b.x ), ceil( point_b.y ) );
	if (camera_motion){
		line( cf, p0, p1, CV_RGB(0,0,0), 2 );
	}
	else{	
		line( cf, p0, p1, CV_RGB(255,255,255), 2 );
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

double OpticalFlow::computeAngle(Point2f point_a, Point2f point_b){
	return  atan2((double) point_b.y - point_a.y, (double) point_b.x - point_a.x) * 180 / PI;
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

bool OpticalFlow::assignBucket(double distance, double angle){
	for(int i = 0; i < buckets.size(); i++){
		if (buckets[i].inBucket(distance, angle)){
			buckets[i].incrementCount();
			return true;
		}
	}
	return false;
}

void OpticalFlow::buildBuckets(double initial_distance, double max_distance, int angle_buckets){
	double increment = 360.0/angle_buckets;
	double last_angle = 0.0;
	double last_distance = 0.0;
	for(double a = increment; a <= 360.0; a+=increment){
		for(double d = initial_distance; d <= max_distance; d+= initial_distance){
			buckets.push_back(Bucket(last_distance, d, last_angle, a));
			last_distance = d;
		}
		last_angle = a;
	}
}

Bucket OpticalFlow::maxBucket(){
	int max=0;
	Bucket max_bucket;
	for (int i = 0; i < buckets.size(); i++){
		if(buckets[i].getCount() > max){
			max = buckets[i].getCount();
			max_bucket = buckets[i];
		}
		buckets[i].resetCount();
	}
	return max_bucket;
}


}



