#include "bball_tracker.h"

#include <iostream>
#include <math.h>
#include <vector>

using namespace cv;

namespace nba_vision {

const char kBinaryWindowName[] = "Bball Segmentation";
// Value represents a probability of two standard deviations for each color.
const double kColorProbThreshold = 0.142625;
// To get rid of object if it's too small.
const int kAreaThreshold = 120;
const double kCircularityThreshold = 0.3;
// Basketball cannot move this far between two frames.
const double kDistanceThreshold = 200;

BballTracker::BballTracker(MultipleKalmanFilter* mkf, bool debug) {
    debug_ = debug;
    if (debug_) {
        namedWindow(kBinaryWindowName, CV_WINDOW_AUTOSIZE);
    }
    mkf_ = mkf;
}

BballTracker::BballTracker(
        MultipleKalmanFilter* mkf,
        const pair<int, int>& init_loc,
        bool debug) {
    debug_ = debug;
    if (debug_) {
        cout << "Initial location: " << init_loc.first << ", " <<
            init_loc.second << endl;
        namedWindow(kBinaryWindowName, CV_WINDOW_AUTOSIZE);
    }
    mkf_ = mkf;
    prediction_ = mkf_->CorrectAndPredictForObject(kBballIndex,
            (Mat_<float>(2, 1) << init_loc.first, init_loc.second));
}

bool area_filter(RegionMetrics* region_metrics) {
    return region_metrics->area < kAreaThreshold;
}

bool circularity_filter(RegionMetrics* region_metrics) {
    return region_metrics->circularity < kCircularityThreshold;
}

void BballTracker::TrackBall(Mat& frame) {
    if (debug_) {
        cout << "Existing prediction: " << prediction_(0) << ", " <<
            prediction_(1) << endl;
    }
    Mat binary_image;
    BballTracker::ColorSegmentation(frame, binary_image);
    Mat components_image;
    int num_components = ComputeConnectedComponents(binary_image, components_image);
    vector<RegionMetrics*> region_metrics_list =
        ComputeRegionMetrics(components_image, num_components);
    // Filter the region_metrics_list by area.
    FilterRegionMetrics(components_image, region_metrics_list, area_filter);
    // Filter the region_metrics_list by circularity.
    FilterRegionMetrics(components_image, region_metrics_list, circularity_filter);
    if (debug_) {
        ConvertComponentsImageToBinary(components_image, binary_image);
        imshow(kBinaryWindowName, binary_image);
    }
    RegionMetrics* region_metrics = FindClosestRegionToPrediction(
            region_metrics_list);
    Mat_<float> new_loc(2, 1);
    if (region_metrics != NULL) {
        double dist = ComputeDistance(
                region_metrics->avg_x, region_metrics->avg_y,
                prediction_(0), prediction_(1));
        if (debug_) {
            cout << "Metrics for bball: " << endl;
            cout << *region_metrics;
            cout << "Distance to prediction: " << dist << endl;
        }
        if (dist < kDistanceThreshold) {
            // Update the prediction with the actual values found in the frame.
            // Otherwise, just use the prediction from the filter because the
            // ball was not correctly found in this frame (it was too far).
            new_loc(0) = region_metrics->avg_x;
            new_loc(1) = region_metrics->avg_y;
            // Draw an orange rectangle where the basketball is.
            int side = sqrt(region_metrics->area);
            int top_left_x = region_metrics->avg_x - side/2;
            int top_left_y = region_metrics->avg_y - side/2;
            rectangle(frame, Rect(top_left_x, top_left_y, side, side),
                    Scalar(255, 0, 165), 2);
        } else {
            new_loc(0) = prediction_(0);
            new_loc(1) = prediction_(1);
        }
    } else {
        new_loc(0) = prediction_(0);
        new_loc(1) = prediction_(1);
    }
    prediction_ = mkf_->CorrectAndPredictForObject(kBballIndex, new_loc);
    // Draw a point for the current prediction.
    circle(frame, Point(prediction_(0), prediction_(1)),
            5, Scalar(255, 0, 0), CV_FILLED, 8, 0);
    if (debug_) {
        cout << "Updated prediction: " << prediction_(0) << ", " <<
            prediction_(1) << endl;
    }
}

RegionMetrics* BballTracker::FindClosestRegionToPrediction(
        vector<RegionMetrics*>& region_metrics_list) {
    double closest = -1;
    RegionMetrics* closest_metrics = NULL;
    for (auto region_metrics : region_metrics_list) {
        double dist = ComputeDistance(
                region_metrics->avg_x, region_metrics->avg_y,
                prediction_(0), prediction_(1));
        if (closest == -1 || dist < closest) {
            closest = dist;
            closest_metrics = region_metrics;
        }
    }
    return closest_metrics;
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
    // Check the ranges for the colors first. Calculate a pseudo probability for that the bball is
    // right color.
    double prob = abs((Phi(R, 110.6875, 10.98134071) - 0.5) * (Phi(G, 74.1875, 9.001518969) - 0.5) *
        (Phi(B, 46.6875, 8.541946134) - 0.5));
    // 0.125 is the max value of prob and the least likely value to be the basketball.
    prob = (0.125 - prob) / 0.125;
    // If all three colors are within two standard deviations, this condition should pass.
    if (prob < kColorProbThreshold) {
        return false;
    }
    // Check the relationships between the colors to further shrink the space.
    double tmp = 0.7618*R - 10.14;
    // Look for two errors out of three to rule out pixel.
    int count = 0;
    if (G > tmp + 7.5 || G < tmp - 7.5) {
        count++;
    }
    if (B > 5*R/8.0 - 45/4.0 || B < R - 80) {
        if (count > 0) {
            return false;
        }
        count++;
    }
    if (B > 5*G/7.0 + 40/7.0 || B < 5*G/4.0 - 255/4.0) {
        if (count > 0) {
            return false;
        }
    }
    // If we get here, its the color of a basketball.
    return true;
}

}

