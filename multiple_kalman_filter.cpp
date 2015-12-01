#include "multiple_kalman_filter.h"

namespace nba_vision {

const int kNumDynamicParams = 4;
const int kNumMeasurementParams = 2;

MultipleKalmanFilter::MultipleKalmanFilter(const int& num_objects,
        const vector< pair<int, int> >* object_locations) {
	kalman_filters_ = map<int, KalmanFilter>();
	for (int i = 0; i < num_objects; ++i) {
		kalman_filters_[i] = KalmanFilter(kNumDynamicParams,
                        kNumMeasurementParams);
		KalmanFilter& kalman_filter = kalman_filters_[i];
		InitKalmanFilter(kalman_filter, (*object_locations)[i].first,
                        (*object_locations)[i].second);
	}
}

Mat MultipleKalmanFilter::CorrectAndPredictForObject(const int& object_idx,
        const Mat_<float>& measurement) {
	if (kalman_filters_.find(object_idx) == kalman_filters_.end()) {
		kalman_filters_[object_idx] = KalmanFilter(kNumDynamicParams,
                        kNumMeasurementParams);
		return InitKalmanFilter(kalman_filters_[object_idx],
                        measurement(0), measurement(1));
	}
	KalmanFilter& kalman_filter = kalman_filters_[object_idx];
	kalman_filter.correct(measurement);
	return kalman_filter.predict();
}

Mat MultipleKalmanFilter::InitKalmanFilter(KalmanFilter& kalman_filter,
        const float& init_x, const float& init_y) {
	// Represents position_x, position_y, velocity_x, velocity_y and how they
        // transition between states.
	kalman_filter.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1,
                0, 0, 1, 0, 0, 0, 0, 1);
	kalman_filter.statePost.at<float>(0) = init_x;
	kalman_filter.statePost.at<float>(1) = init_y;
	kalman_filter.statePost.at<float>(2) = 0;
	kalman_filter.statePost.at<float>(3) = 0;
	setIdentity(kalman_filter.measurementMatrix);
	setIdentity(kalman_filter.processNoiseCov, Scalar::all(1e-6));
	setIdentity(kalman_filter.measurementNoiseCov, Scalar::all(1e-3));
	setIdentity(kalman_filter.errorCovPost, Scalar::all(.01));
	return kalman_filter.predict();
}

}
