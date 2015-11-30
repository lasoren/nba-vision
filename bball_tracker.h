#include <opencv2/highgui/highgui.hpp>

#include "multiple_kalman_filter.h"

using namespace std;
using namespace cv;

namespace nba_vision {

const int kBballIndex = 0;

class BballTracker {
public:
    explicit BballTracker(const MultipleKalmanFilter* mkf);

    // Initialize the tracker with a starting location.
    BballTracker(const MultipleKalmanFilter* mkf, const pair<int, int>& init_loc);
    
    // Performs color segmentation, connected components, circularity, size filtering,
    // finds basketball using prediction from Kalman filter or where it should be (if
    // it is hidden) and then draws the location of the ball on the frame.
    TrackBall(Mat& frame);

private:
    // Segments the image into background and foreground by finding pixels in the
    // range of the color of the ball.
    ColorSegmentation(const Mat& frame, Mat& binary_image) const;
    
    // Returns true if the color could be that of the basketball. False otherwise.
    bool IsBballColor(const Vec3b& color) const;

    // A pointer to the MultipleKalmanFilter object owned by calling program.
    const MultipleKalmanFilter* mkf_; 
}

}
