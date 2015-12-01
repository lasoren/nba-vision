#include "bball_tracker.h"

using namespace cv;

namespace nba_vision {

const char kBinaryWindowName[] = "Bball Segmentation";

BballTracker::BballTracker(MultipleKalmanFilter* mkf, bool debug) {
    debug_ = debug;
    if (debug_) {
        namedWindow(kBinaryWindowName, CV_WINDOW_AUTOSIZE);
    }
    mkf_ = mkf;
}

BballTracker::BballTracker(
        MultipleKalmanFilter* mkf,
        const pair<int, int>& init_loc) {
    mkf_ = mkf;
    mkf_->CorrectAndPredictForObject(kBballIndex,
            (Mat_<float>(2, 1) << init_loc.first, init_loc.second));
}

void BballTracker::TrackBall(Mat& frame) {
    Mat binary_image;
    BballTracker::ColorSegmentation(frame, binary_image);
    if (debug_) {
        imshow(kBinaryWindowName, binary_image);
    }
}

void BballTracker::ColorSegmentation(const Mat& frame, Mat& binary_image) const {
    // Apply color rules to segment out the basketball from the frame.
    binary_image = Mat::zeros(frame.rows, frame.cols, CV_8UC1); 
    for (int r = 0; r < frame.rows; r++) {
        for (int c = 0; c < frame.cols; c++) {
            if (IsBballColor(frame.at<Vec3b>(r, c))) {
                binary_image.at<uchar>(r, c) = 255;
            }
        }
    }
}

bool BballTracker::IsBballColor(const Vec3b& color) const {
    // Determines whether a color is potentially that of a basketball.
    int R, G, B;
    B = color[0]; G = color[1]; R = color[2];
    // Check the ranges for the colors first.
    if (R > 135 || R < 87) {
        return false;
    }
    if (G > 99 || G < 51) {
        return false;
    }
    if (B > 70 || B < 27) {
        return false;
    }
    // Check the relationships between the colors to further shrink the space.
    double tmp = 0.7618*R - 10.14;
    if (G > tmp + 7.5 || G < tmp - 7.5) {
        return false;
    }
    if (B > 5*R/8.0 - 45/4.0 || B < R - 80) {
        return false;
    }
    if (B > 5*G/7.0 + 40/7.0 || B < 5*G/4.0 - 255/4.0) {
        return false;
    }
    // If we get here, its the color of a basketball.
    return true;
}

}

