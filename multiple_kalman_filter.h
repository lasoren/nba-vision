#ifndef MULTIPLE_KALMAN_FILTER_H
#define MULTIPLE_KALMAN_FILTER_H

#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;

namespace nba_vision {

// An extension of the opencv KalmanFilter, to perform kalman filtering on multiple objects.
class MultipleKalmanFilter {

public:
	// Initialize with a number of objects and object locations.
	MultipleKalmanFilter(const int& num_objects, const vector< pair<int, int> >* object_locations);

	// Update existing objects or create a new object, with a new measurement.
	Mat CorrectAndPredictForObject(const int& object_idx, const Mat_<float>& measurement);

private:
	// Internal method for initializing the opencv KalmanFilter.
	Mat InitKalmanFilter(KalmanFilter& kalman_filter, const float& init_x, const float& init_y);

	map<int, KalmanFilter> kalman_filters_;

};

}

#endif  // MULTIPLE_KALMAN_FILTER_H
