#ifndef BBALL_TRACKER_H
#define BBALL_TRACKER_H

#include <opencv2/highgui/highgui.hpp>

#include "multiple_kalman_filter.h"
#include "util.h"

using namespace std;
using namespace cv;

#define DEFAULT 0
#define SHOT 1

namespace nba_vision {

class BballTracker {
public:
    BballTracker(MultipleKalmanFilter* mkf, bool debug=false);

    // Initialize the tracker with a starting location.
    BballTracker(MultipleKalmanFilter* mkf, const pair<int, int>& init_loc,
            bool debug=false);
    
    // Performs color segmentation, connected components, circularity, size filtering,
    // finds basketball using prediction from Kalman filter or where it should be (if
    // it is hidden) and then draws the location of the ball on the frame.
    void TrackBall(Mat& frame);

private:
    // Loops through the region_metrics_list and finds the region closest to the
    // prediction.
    RegionMetrics* FindClosestRegionToPrediction(
            vector<RegionMetrics*>& region_metrics_list);

    // Segments the image into background and foreground by finding pixels in the
    // range of the color of the ball.
    void ColorSegmentation(const Mat& frame, Mat& binary_image) const;
    
    // Returns true if the color could be that of the basketball. False otherwise.
    bool IsBballColor(const Vec3b& color) const;

    // Uses template matching algorithm to find the net in the frame and
    // draws a rectangle around it.Returns true if the net was found in
    // the current frame and rect is not null. False otherwise.
    static bool FindNet(Mat& detect, Rect* rect);

    void InitNetTemplate();

    static void LoadAndCreateEdgesTemplate(const char* filename, Mat& edges);

    // A pointer to the MultipleKalmanFilter object owned by calling program.
    MultipleKalmanFilter* mkf_; 
    // Display debug output.
    bool debug_;
    // Save the predicted values from the kalman filter.
    Mat_<float> prediction_;
    // Saves the state of the ball.
    int state_;
    // Stores the template edges for the net template.
    static unique_ptr<Mat> template_edges_;
    static unique_ptr<Point> prev_net_location_;
    static int prev_net_width_;
    static int prev_net_height_;
};

}

#endif  // BBALL_TRACKER_H
