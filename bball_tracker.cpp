#include "bball_tracker.h"

#include <iostream>
#include <math.h>
#include <vector>

using namespace cv;

namespace nba_vision {
// For the multiple kalman filter. A unique integer to identify the ball.
const int kBballIndex = 0;

const char kBinaryWindowName[] = "Bball Segmentation";
// Value represents a probability of two standard deviations for each color.
const double kColorProbThreshold = 0.142625;
// To get rid of object if it's too small.
const int kAreaThreshold = 120;
const double kCircularityThreshold = 0.3;
// Basketball cannot move this far between two frames.
const double kDistanceThreshold = 200;
// Distance between location and prediction for it to be added to path.
const double kTighterDistanceThreshold = 50;
// The location of the template for a net.
const char kNetTemplateFilename[] = "metadata/net_template.jpg";
const char kNetTemplateWindowName[] = "Net Template Edges";
// Net cannot move this far between frames.
const double kNetDistanceThreshold = 100;

const double kMaxScale = 0.2;

unique_ptr<Mat> BballTracker::template_edges_ = nullptr;
unique_ptr<Point> BballTracker::prev_net_location_ = nullptr;
int BballTracker::prev_net_width_ = 0;
int BballTracker::prev_net_height_ = 0;

void BballTracker::InitNetTemplate() {
    if (template_edges_ == nullptr) {
        template_edges_.reset(new Mat());
        LoadAndCreateEdgesTemplate(kNetTemplateFilename, *template_edges_);
        if (debug_) {
            namedWindow(kNetTemplateWindowName, CV_WINDOW_AUTOSIZE);
            imshow(kNetTemplateWindowName, *template_edges_);
        }
    }
}

BballTracker::BballTracker(MultipleKalmanFilter* mkf, bool debug) {
    debug_ = debug;
    if (debug_) {
        namedWindow(kBinaryWindowName, CV_WINDOW_AUTOSIZE);
    }
    InitNetTemplate();
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
    InitNetTemplate();
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
    // Find the hoop.
    Rect rect;
    bool found_net = FindNet(frame, rect);

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
    double dist = -1;
    if (region_metrics != NULL) {
        dist = ComputeDistance(
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
                    Scalar(0, 165, 255), 2);
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
            5, Scalar(255, 255, 255), CV_FILLED, 8, 0);
    if (debug_) {
        cout << "Updated prediction: " << prediction_(0) << ", " <<
            prediction_(1) << endl;
    }
    // Update state of the ball based on its location.
    if (found_net) { 
         UpdateBallState(rect, new_loc);
    }

    if (state_ == SHOT) {
        if (dist != -1 && dist < kTighterDistanceThreshold) {
            AddLocationToPath(pair<int, int>(new_loc(0), new_loc(1)));
        }
        DrawPath(frame);
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

bool BballTracker::FindNet(Mat& detect, Rect& rect) {
    Mat detect_templ;
    // Assume that the net will be on the top half of the image.
    Mat detect_portion = detect(
            cv::Range(0, detect.rows / 2), cv::Range(0, detect.cols));
    // Convert to greyscale.
    cvtColor(detect_portion, detect_templ, CV_BGR2GRAY); 
    // Find edges in the template image, using Canny edge detection algorithm.
    Canny(detect_templ, detect_templ, 120, 300, 3);

    Mat resized_templ; 
    Mat result;
    // Compute max scale value.
    double max_scale = detect_templ.size().height /
        (double) template_edges_->size().height; 
    if (max_scale > kMaxScale) {
        max_scale = kMaxScale;
    }
    double max_correlation_value = 0.0;
    Point max_correlation_location; 
    double max_correlation_scalar = 0.0;
    for (double scale = max_scale; scale > 0.1; scale -= 0.05) {
        resize(*template_edges_, resized_templ, Size(), scale, scale);
        // Make sure the resized template does not exceed the frame size width.
        if (resized_templ.size().width > detect_templ.size().width) {
            continue;
        }
        // Perform correlation coefficient template matching.
        matchTemplate(detect_templ, resized_templ, result, CV_TM_CCOEFF);
        
        double current_max_value; Point current_max_location;
        // Get the maximum value and its location.
        minMaxLoc(result, NULL,
                  &current_max_value, NULL, &current_max_location);

        // Record new globabl maximum if found.
        if (current_max_value > max_correlation_value) {
            max_correlation_value = current_max_value;
            max_correlation_location = current_max_location;
            max_correlation_scalar = scale;
        }
    }

    int height = template_edges_->rows * max_correlation_scalar;
    int width = template_edges_->cols * max_correlation_scalar;
    if (prev_net_location_ == nullptr) {
        prev_net_location_.reset(
                new Point(max_correlation_location.x,
                    max_correlation_location.y));
        return false;
    } else {
        rect = Rect(prev_net_location_->x, prev_net_location_->y,
                prev_net_width_, prev_net_height_);
        rectangle(detect, rect, Scalar(0, 0, 128), 2);
        if (ComputeDistance(
                    max_correlation_location.x,
                    max_correlation_location.y,
                    prev_net_location_->x,
                    prev_net_location_->y) < kNetDistanceThreshold) {
            // Update the location of the net.
            prev_net_location_->x = max_correlation_location.x;
            prev_net_location_->y = max_correlation_location.y;
            prev_net_width_ = width;
            prev_net_height_ = height;
            rect = Rect(prev_net_location_->x, prev_net_location_->y, width, height);
            rectangle(detect, rect, Scalar(0, 0, 255), 2);
        }
    }
    return true;
}

void BballTracker::LoadAndCreateEdgesTemplate(
        const char* filename, Mat& edges) {
    edges = imread(filename);
    if (!edges.data) {
        cout << "Could not find or open image: " << filename << endl;
        return;
    }
    // Convert to greyscale.
    cvtColor(edges, edges, CV_BGR2GRAY); 
    // Find edges in the template image, using Canny edge detection algorithm.
    Canny(edges, edges, 100, 200, 3, true);
}

void BballTracker::UpdateBallState(const Rect& net_rect,
        const Mat_<float>& current_loc) {
    switch (state_) {
        case SHOT: {
            int rect_bottom = net_rect.y + net_rect.height;
            if (current_loc(1) > rect_bottom && prediction_(1) > rect_bottom) {
                if (debug_) {
                    cout << "Basketball state changed to DEFAULT." << endl;
                }
                state_ = DEFAULT;
            }
            break;
        }
        default: {
            // Check that both the current location and the prediction are
            // above the net. Then, assume the ball has been shot.
            if (current_loc(1) < net_rect.y && prediction_(1) < net_rect.y) {
                if (debug_) {
                    cout << "Basketball state changed to SHOT." << endl;
                }
                state_ = SHOT;
            }
            break;
        }
    }
}

void BballTracker::AddLocationToPath(const pair<int, int>& location) {
    path_.push_back(location);
    // Delete the first element of the path if the size of the path gets too large.
    if (path_.size() > PATH_SIZE) {
        path_.erase(path_.begin());
    }
}

void BballTracker::DrawPath(Mat& frame) {
    int counter = 0;
    for (int i = path_.size() - 1; i >= 1 && counter < PATH_SIZE; i--) {
        line(frame, Point(path_[i].first, path_[i].second),
                Point(path_[i - 1].first, path_[i - 1].second),
                Scalar(0, 165, 255), 2);
        counter++;
    }
}

}

