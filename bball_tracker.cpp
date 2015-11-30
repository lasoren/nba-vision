#include "bball_tracker.h"

using namespace cv;

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

BballTracker::TrackBall(Mat& frame) {

}

BballTracker::ColorSegmentation(const Mat& frame, Mat& binary_image) const {
    // Apply color rules to segment out the basketball from the frame.
    binary_image = Mat::zeros(frame.rows, frame.cols, CV_8UC1); 
    for (int r = 0; r < frame.rows; r++) {
        for (int c = 0; c < frame.cols; c++) {
            
        }
    }
}

bool BballTracker::IsBballColor(const Vec3b& color) const {
}

}

