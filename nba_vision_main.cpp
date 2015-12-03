#include <iostream>
#include <mutex>

#include "opencv2/highgui/highgui.hpp"

#include "multiple_kalman_filter.h"
#include "bball_tracker.h"

using namespace cv;
using namespace std;
using namespace nba_vision;

const bool kDebug = true;
const char kWindowName[] = "Output";

// Whether or not the user has clicked on the location of the ball in the
// first frame.
bool ball_init;
// The starting location of the ball.
int init_ball_x, init_ball_y;
// Used for allowing the user to click on the starting position of the ball.
void MouseCallBack(int event, int x, int y, int flags, void* userdata);
// For locking and unlocking global variables from the UI thread.
mutex mtx;

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
    unique_ptr<BballTracker> bball_tracker;
    ball_init = false;

    while (true) {
        Mat frame;
        bool success = video_capture.read(frame);
        
        if (!success) {
            cout << "Cannot read current frame from the video file." << endl;
            break;
        }

        imshow(kWindowName, frame);
        mtx.lock();
        if (!ball_init) {
            cout << "Click on the basketball to set its initial location." << endl;
            setMouseCallback(kWindowName, MouseCallBack, NULL);
        }
        while (!ball_init) {
            mtx.unlock();
            if (waitKey(30) == 27) {
                cout << "Esc key pressed." << endl;
                break;
            }
            mtx.lock();
        }
        if (bball_tracker == nullptr) {
            // Initialize the bball with the location of the mouse click.
            bball_tracker.reset(new BballTracker(&mkf,
                        pair<int, int>(init_ball_x, init_ball_y),
                        kDebug));
        }
        mtx.unlock();
        // Track the basketball in each frame.
        bball_tracker->TrackBall(frame);
        

        if (waitKey(20) == 27) {
            cout << "Esc key pressed." << endl;
            break;
        }
    }

    return 0;
}

void MouseCallBack(int event, int x, int y, int flags, void* userdata) {
    if  (event == EVENT_LBUTTONDOWN) {
        mtx.lock();
        if (!ball_init) {
            ball_init = true;
            init_ball_x = x;
            init_ball_y = y;
        }
        mtx.unlock();
    }
}
