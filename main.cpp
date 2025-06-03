#include <iostream>
#include <opencv2/opencv.hpp>
#include "diverse.h"

using namespace std;
using namespace cv;

int main() {

    Mat image = imread(R"(C:\Users\radua\CLionProjects\Proiect_PI\Red_Eye8.jpg)");

    imshow("image", image);

    // Testare break_channels
    image_channels_bgr bgr_channels = break_channels(image);
    //display_channels(bgr_channels);

    // Testare bgr_2_grayscale
    Mat grayscale = bgr_2_grayscale(image);
    //imshow("Grayscale", grayscale);

    // Testare grayscale_2_binary
    Mat binary = grayscale_2_binary(grayscale, 128);
    //imshow("Binary", binary);

    // Testare bgr_2_hsv
    image_channels_hsv hsv_channels = bgr_2_hsv(bgr_channels);
    //display_hsv_channels(hsv_channels);

    // Testare masca rosie
    Mat redMask = create_red_mask(hsv_channels);
    imshow("Red Eye Mask", redMask);

    // Testare fill_holes
    Mat redMask_filled = fill_holes(redMask);
    imshow("Red Eye Mask - Holes Filled", redMask_filled);

    // Testare labeling
    labels two_pass_label = Two_pass_labeling(redMask);
    Mat result_two_pass = color_labels(two_pass_label);
    imshow("Two pass", result_two_pass);
    cout << "Obiecte detectate: " << two_pass_label.no_labels << endl;

    // Testare functie de extragere a ochilor
    Mat circular_objects = extract_circular_objects(redMask);
    imshow("Circular Objects (Eyes)", circular_objects);

    // Imagine finala
    Mat corrected_image = correct_red_eyes(image, circular_objects);
    imshow("Corrected Image", corrected_image);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
