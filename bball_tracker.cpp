#include "bball_tracker.h"

namespace nba_vision {

BballTracker::BballTracker(const MultipleKalmanFilter* mkf) {
    mkf_ = mkf;
}

BballTracker::BballTracker(
        const MultipleKalmanFilter* mkf,
        const pair<int, int>& init_loc) {
    mkf_ = mkf;
    mkf.CorrectAndPredictForObject(kBballIndex,
            (Mat_<float>(2, 1) << init_loc.first, init_loc.second));
}

BballTracker::TrackBall(Mat frame) {

}

}

