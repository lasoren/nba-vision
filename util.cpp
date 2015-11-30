#include "util.h"

using namespace cv;

namespace nba_vision {

bool IsBoundaryPixel(const PixelLoc& current_pixel, const int& component_index,
        const Mat& components_image) {
	// Check to see if pixel is surrounded entirely by pixels of the same
	// component. If so, return false.
	for (int i = -1; i <= 1; ++i) {
		for (int j = -1; j <= 1; ++j) {
			int new_r = current_pixel.first + i;
			int new_c = current_pixel.second + j;
			// Make sure the new pixel location isn't out of bounds of
                        // the array.
			if (new_r < 0 || new_r >= components_image.rows || new_c < 0
                                || new_c >= components_image.cols) {
				continue;
			}
			if (components_image.at<uchar>(new_r, new_c) !=
                                component_index) {
				return true;
			}
		}
	}
	// All surrounding pixels are of the same component. It is an inner pixel.
	return false;
}

void ComputeEminEmax(const double& a, const double& b, const double& c,
	double* e_min, double* e_max) {
	*e_min = (a + c)/2 - (((a - c)/2) * (a - c)/(sqrt(pow(a - c, 2) + pow(b, 2))))
            - (b / 2) * (b / (sqrt(pow(a - c, 2) + pow(b, 2))));
	*e_max = (a + c)/2 - (((a - c)/2) * -(a - c) / (sqrt(pow(a - c, 2) +
            pow(b, 2)))) - (b / 2) * (-b / (sqrt(pow(a - c, 2) + pow(b, 2))));
}

vector<RegionMetrics*> ComputeRegionMetrics(const Mat& components_image,
        const int& num_components) {
	// Computes the area, orientation, and circularity.
	// Also, identify and count the boundary pixels of each region,
	// and compute compactness, the ratio of the area to the perimeter.
	vector<RegionMetrics*> region_metrics_list = vector<RegionMetrics*>();
	for (int i = 0; i < num_components; ++i) {
		region_metrics_list.emplace_back(new RegionMetrics());
	}

	for (int i = 0; i < components_image.rows; ++i) {
		for (int j = 0; j < components_image.cols; ++j) {
			int component_index = components_image.at<uchar>(i, j);
			if (component_index != 1) {
				RegionMetrics* region_metrics =
                                    region_metrics_list[component_index - 2];
				region_metrics->component_index = component_index;
				region_metrics->area += 1;
				if (IsBoundaryPixel(PixelLoc(i, j), component_index,
                                            components_image)) {
					region_metrics->num_boundary_pixels += 1;
				}
				region_metrics->avg_x += j;
				region_metrics->avg_y += i;
			}
		}
	}

	for (auto region_metrics : region_metrics_list) {
		// Compute more metrics post-parsing the photo.
		region_metrics->area_perimeter_ratio =
			region_metrics->area / (double)
                        region_metrics->num_boundary_pixels;
		region_metrics->avg_x = region_metrics->avg_x / region_metrics->area;
		region_metrics->avg_y = region_metrics->avg_y / region_metrics->area;
		region_metrics->compactness = pow(region_metrics->num_boundary_pixels,
                        2) / region_metrics->area;
	}

	// Second pass through the image to compute more properties.
	for (int i = 0; i < components_image.rows; ++i) {
		for (int j = 0; j < components_image.cols; ++j) {
			int component_index = components_image.at<uchar>(i, j);
			if (component_index != 1) {
				RegionMetrics* region_metrics = region_metrics_list[
                                    component_index - 2];
				region_metrics->x_second_moment +=
                                    pow(j - region_metrics->avg_x, 2);
				region_metrics->y_second_moment +=
                                    pow(i - region_metrics->avg_y, 2);
				region_metrics->cross_second_moment +=
                                    (j - region_metrics->avg_x) *
                                    (i - region_metrics->avg_y);
			}
		}
	}

	for (auto region_metrics : region_metrics_list) {
		// Compute more metrics post-parsing the photo.
		region_metrics->x_second_moment = region_metrics->x_second_moment /
                    region_metrics->area;
		region_metrics->y_second_moment = region_metrics->y_second_moment /
                    region_metrics->area;
		region_metrics->cross_second_moment =
			region_metrics->cross_second_moment /
                        region_metrics->area;
		// Compute orientation and circularity.
		region_metrics->orientation = atan(2 *
                        region_metrics->cross_second_moment /
			(region_metrics->x_second_moment -
                         region_metrics->y_second_moment)) / 2;
		double e_min, e_max;
		ComputeEminEmax(region_metrics->x_second_moment,
			2 * region_metrics->cross_second_moment,
			region_metrics->y_second_moment,
			&e_min, &e_max);
		region_metrics->circularity = e_min / e_max;
	}
	return region_metrics_list;
}

void SearchForObject(const Mat& binary_image, const PixelLoc& starting_point,
        Mat& output_image) {
	stack<PixelLoc> pixel_stack;
	pixel_stack.push(starting_point);
	while (!pixel_stack.empty()) {
		PixelLoc current_pixel = pixel_stack.top();
		pixel_stack.pop();
		// Find this pixels neighbors.
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				int new_r = current_pixel.first + i;
				int new_c = current_pixel.second + j;
				// Make sure the new pixel location isn't out of
                                // bounds of the array.
				if (new_r < 0 || new_r >= output_image.rows ||
                                        new_c < 0 || new_c >= output_image.cols) {
					continue;
				}
				// Make sure the new pixel location isn't already
                                // labeled.
				if (output_image.at<uchar>(new_r, new_c) != 0) {
					continue;
				}
				// If this pixel value in the binary image is 0, set
                                // the output_image to component 1.
				if (binary_image.at<uchar>(new_r, new_c) == 0) {
					output_image.at<uchar>(new_r, new_c) = 1;
				}
				else {  // Otherwise, set this pixel to the same
                                        // component as the current_pixel and add
                                        // this pixel to the stack.
					output_image.at<uchar>(new_r, new_c) =
                                            output_image.at<uchar>(current_pixel.first,
                                                    current_pixel.second);
					pixel_stack.push(PixelLoc(new_r, new_c));
				}
			}
		}
	}
}

int ComputeConnectedComponents(const Mat& binary_image, Mat output_image) {
	output_image = Mat::zeros(binary_image.rows, binary_image.cols, CV_8UC1);
	// 1 will mean not part of an object (part of background).
	int current_component_label = 2;
	// Loop through binary image and find non-zero pixels (binary 1s) and then
        // search for the object.
	for (int r = 0; r < binary_image.rows; r++) {
		for (int c = 0; c < binary_image.cols; c++) {
			// Keep going if the pixel is already labeled.
			if (output_image.at<uchar>(r, c) != 0) {
				continue;
			}
			// If binary image is 0, label it component 1 to mark that
                        // it has been looked at.
			if (binary_image.at<uchar>(r, c) == 0) {
				output_image.at<uchar>(r, c) = 1;
			}
			else {  // If the binary image value here is high, search for
                                // the corresponding object.
				output_image.at<uchar>(r, c) = current_component_label;
				// Recursively search for this entire object based on
                                // neighboring pixels.
				SearchForObject(binary_image, PixelLoc(r, c),
                                        output_image);
				// Increment the current component label so that
                                // the next object gets a new label.
				current_component_label++;
			}
		}
	}
	// Entire output_image should be labeled now. 1 for background. 2 - N for
        // the other objects.
	return current_component_label - 1;
}

}
