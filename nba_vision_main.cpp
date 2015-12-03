#include <iostream>

#include "opencv2/highgui/highgui.hpp"

#include "multiple_kalman_filter.h"
#include "bball_tracker.h"

using namespace cv;
using namespace std;
using namespace nba_vision;

const bool kDebug = true;
const char kWindowName[] = "Output";

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "usage: " << argv[0] << " <filename>" << endl;
        return -1;
    }
    // Open the specified video file.
    VideoCapture video_capture(argv[1]);

    if (!video_capture.isOpened()) {
        cout << "Cannot open the video file." << endl;
        return -1;
    }

    namedWindow(kWindowName, CV_WINDOW_AUTOSIZE);

    MultipleKalmanFilter mkf(0, NULL);
    BballTracker bball_tracker(&mkf, kDebug);

    while(true) {
        Mat frame;
        bool success = video_capture.read(frame);
        
        if (!success) {
            cout << "Cannot read current frame from the video file." << endl;
            break;
        }
        // Track the basketball in each frame.
        bball_tracker.TrackBall(frame);
        
        imshow(kWindowName, frame);

        if (waitKey(5) == 27) {
            cout << "Esc key pressed." << endl;
            break;
        }
    }

    return 0;
}
