#ifndef UTIL_H
#define UTIL_H

#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

namespace nba_vision {

class RegionMetrics {
public:
	int component_index;
	double area;
	int num_boundary_pixels;
	double area_perimeter_ratio;
	double avg_x;
	double avg_y;
	double x_second_moment;
	double y_second_moment;
	double cross_second_moment;

	double orientation;
	double circularity;
	double compactness;

	RegionMetrics() {
		component_index = 0;
		area = 0;
		num_boundary_pixels = 0;
		area_perimeter_ratio = 0;
		avg_x = 0;
		avg_y = 0;
		x_second_moment = 0;
		y_second_moment = 0;
		cross_second_moment = 0;

		orientation = 0;
		circularity = 0;
		compactness = 0;
	}

	// For easy printing of region_metrics.
	friend ostream &operator<<(ostream &output, const RegionMetrics region_metrics) {
		output << "Region metrics for component: " << region_metrics.component_index << endl;
		output << "  Area: " << region_metrics.area << endl;
		output << "  Number of boundary pixels: " << region_metrics.num_boundary_pixels << endl;
		output << "  Area-perimeter ratio: " << region_metrics.area_perimeter_ratio << endl;
		output << "  Average X: " << region_metrics.avg_x << endl;
		output << "  Average Y: " << region_metrics.avg_y << endl;
		output << "  Second moment X: " << region_metrics.x_second_moment << endl;
		output << "  Second moment Y: " << region_metrics.y_second_moment << endl;
		output << "  Cross second moment: " << region_metrics.cross_second_moment << endl;
		output << "  Orientation: " << region_metrics.orientation << endl;
		output << "  Circularity: " << region_metrics.circularity << endl;
		output << "  Compactness: " << region_metrics.compactness << endl;
		output << endl << endl;
		return output;
	}
};

// Given a binary segmented image, find all distinct objects and assign labels
// to those objects.
int ComputeConnectedComponents(const Mat& binary_image, Mat& output_image);

// Computes all of the metrics for the output_image from ComputeConnectedComponents.
vector<RegionMetrics*> ComputeRegionMetrics(const Mat& components_image, const int& num_components);

}

#endif  // UTIL_H
