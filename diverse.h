#ifndef DIVERSE_H
#define DIVERSE_H
#define PI 3.14159265
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

typedef struct {
    Mat B;
    Mat G;
    Mat R;
} image_channels_bgr;

typedef struct {
    Mat H;
    Mat S;
    Mat V;
} image_channels_hsv;

typedef struct {
    int size;
    vector<int> di;
    vector<int> dj;
} neighborhood_structure;

typedef struct {
    Mat labels;
    int no_labels;
}labels;

struct perimeter {
    Mat contour;
    int length;
};

image_channels_bgr break_channels(Mat source);

void display_channels(image_channels_bgr bgr_channels);

Mat bgr_2_grayscale(Mat source);

Mat grayscale_2_binary(Mat source, int threshold);

image_channels_hsv bgr_2_hsv(image_channels_bgr bgr_channels);

void display_hsv_channels(image_channels_hsv hsv_channels);

bool IsInside(Mat img, int i, int j);

Mat erosion(Mat source, neighborhood_structure neighborhood, int no_iter);

Mat dilation(Mat source, neighborhood_structure neighborhood, int no_iter);

Mat create_red_mask(image_channels_hsv hsv_channels);

Mat color_labels(labels labels_str);

labels Two_pass_labeling(Mat source);

Mat get_object_instance(Mat source, Vec3b color);

perimeter naive_perimeter(Mat binary_object);

int compute_area(Mat binary_object);

float compute_thinness_ratio(int area, int perimeter);

Mat extract_circular_objects(Mat source);

Mat correct_red_eyes(Mat original_image, Mat circular_objects);

#endif //DIVERSE_H
