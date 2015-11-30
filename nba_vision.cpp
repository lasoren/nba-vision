#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

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
    while(true) {
        Mat frame;
        bool success = video_capture.read(frame);
        
        if (!success) {
            cout << "Cannot read current frame from the video file." << endl;
            break;
        }
        
        imshow(kWindowName, frame);

        if (waitKey(30) == 27) {
            cout << "Esc key pressed." << endl;
            break;
        }
    }

    return 0;
}
