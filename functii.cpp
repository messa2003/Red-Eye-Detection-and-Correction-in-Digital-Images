#include <iostream>
#include <opencv2/opencv.hpp>
#include "diverse.h"
using namespace std;
using namespace cv;

image_channels_bgr break_channels(Mat source) {
    int rows = source.rows, cols = source.cols;
    image_channels_bgr bgr_channels;

    Mat Blue = Mat(rows, cols, CV_8UC1);
    Mat Green = Mat(rows, cols, CV_8UC1);
    Mat Red = Mat(rows, cols, CV_8UC1);

    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
            Vec3b pixel = source.at<Vec3b>(i, j);

            Blue.at<uchar>(i, j) = pixel[0];
            Green.at<uchar>(i, j) = pixel[1];
            Red.at<uchar>(i, j) = pixel[2];
        }
    }
    bgr_channels.B = Blue;
    bgr_channels.G = Green;
    bgr_channels.R = Red;

    return bgr_channels;
}

void display_channels(image_channels_bgr bgr_channels) {
    int rows = bgr_channels.B.rows;
    int cols = bgr_channels.B.cols;

    Mat blueImage(rows, cols, CV_8UC3, Scalar(0, 0, 0));
    Mat greenImage(rows, cols, CV_8UC3, Scalar(0, 0, 0));
    Mat redImage(rows, cols, CV_8UC3, Scalar(0, 0, 0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            blueImage.at<Vec3b>(i, j)[0] = bgr_channels.B.at<uchar>(i, j);
            blueImage.at<Vec3b>(i, j)[1] = 0;
            blueImage.at<Vec3b>(i, j)[2] = 0;

            greenImage.at<Vec3b>(i, j)[0] = 0;
            greenImage.at<Vec3b>(i, j)[1] = bgr_channels.G.at<uchar>(i, j);
            greenImage.at<Vec3b>(i, j)[2] = 0;

            redImage.at<Vec3b>(i, j)[0] = 0;
            redImage.at<Vec3b>(i, j)[1] = 0;
            redImage.at<Vec3b>(i, j)[2] = bgr_channels.R.at<uchar>(i, j);
        }
    }

    imshow("Blue Channel", blueImage);
    imshow("Green Channel", greenImage);
    imshow("Red Channel", redImage);
}

Mat bgr_2_grayscale(Mat source) {
    Mat grayscale_image(source.rows, source.cols, CV_8UC1);

    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
            Vec3b pixel = source.at<Vec3b>(i, j);
            unsigned char B = pixel[0];
            unsigned char G = pixel[1];
            unsigned char R = pixel[2];
            unsigned char gray = (B + G + R) / 3;
            grayscale_image.at<unsigned char>(i, j) = gray;
        }
    }

    return grayscale_image;
}

Mat grayscale_2_binary(Mat source, int threshold) {
    int rows = source.rows, cols = source.cols;
    Mat binary(rows, cols, CV_8UC1);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            unsigned char pixel = source.at<unsigned char>(i, j);
            binary.at<unsigned char>(i, j) = (pixel < threshold) ? 0 : 255;
        }
    }

    return binary;
}

image_channels_hsv bgr_2_hsv(image_channels_bgr bgr_channels) {
    int rows = bgr_channels.B.rows;
    int cols = bgr_channels.B.cols;

    Mat H = Mat::zeros(rows, cols, CV_32FC1);
    Mat S = Mat::zeros(rows, cols, CV_32FC1);
    Mat V = Mat::zeros(rows, cols, CV_32FC1);
    image_channels_hsv hsv_channels;

    float M, m, C;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float r = static_cast<float>(bgr_channels.R.at<uchar>(i, j)) / 255;
            float g = static_cast<float>(bgr_channels.G.at<uchar>(i, j)) / 255;
            float b = static_cast<float>(bgr_channels.B.at<uchar>(i, j)) / 255;

            M = std::max({r, g, b});
            m = std::min({r, g, b});
            C = M - m;

            V.at<float>(i, j) = M;

            if (M != 0.0f)
                S.at<float>(i, j) = C / M;
            else
                S.at<float>(i, j) = 0.0f;

            if (C != 0.0f) {
                if (M == r)
                    H.at<float>(i, j) = 60.0f * (g - b) / C;
                else if (M == g)
                    H.at<float>(i, j) = 120.0f + 60.0f * (b - r) / C;
                else
                    H.at<float>(i, j) = 240.0f + 60.0f * (r - g) / C;
            } else {
                H.at<float>(i, j) = 0.0f;
            }

            if (H.at<float>(i, j) < 0.0f)
                H.at<float>(i, j) += 360.0f;
        }
    }

    hsv_channels.H = H;
    hsv_channels.S = S;
    hsv_channels.V = V;

    return hsv_channels;
}

void display_hsv_channels(image_channels_hsv hsv_channels) {
    Mat H_norm, S_norm, V_norm;

    normalize(hsv_channels.H, H_norm, 0, 255, NORM_MINMAX, CV_8UC1);
    normalize(hsv_channels.S, S_norm, 0, 255, NORM_MINMAX, CV_8UC1);
    normalize(hsv_channels.V, V_norm, 0, 255, NORM_MINMAX, CV_8UC1);

    imshow("H", H_norm);
    imshow("S", S_norm);
    imshow("V", V_norm);
}

bool IsInside(Mat img, int i, int j) {
    if (i < 0 || i >= img.rows || j < 0 || j >= img.cols) {
        return false;
    }
    return true;
}

Mat erosion(Mat source, int no_iter) {
    Mat dst = source.clone();
    Mat aux;
    int rows = source.rows;
    int cols = source.cols;
    int dx[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    int dy[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

    for (int iter = 0; iter < no_iter; iter++) {
        aux = dst.clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (dst.at<uchar>(i, j) == 255) { // Daca pixelul este alb
                    bool all_inside = true;
                    for (int k = 0; k < 8; k++) {
                        int ni = i + dx[k];
                        int nj = j + dy[k];
                        if (!IsInside(source, ni, nj) || aux.at<uchar>(ni, nj) == 0) {
                            all_inside = false;
                            break;
                        }
                    }
                    if (!all_inside) {
                        dst.at<uchar>(i, j) = 0; // Eroziune: pixelul devine negru daca nu sunt albi toți vecinii
                    }
                }
            }
        }
    }
    return dst;
}

Mat dilation(Mat source, int no_iter) {
    Mat dst = source.clone();
    Mat aux;
    int rows = source.rows;
    int cols = source.cols;
    int dx[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    int dy[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

    for (int iter = 0; iter < no_iter; iter++) {
        aux = dst.clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (dst.at<uchar>(i, j) == 0) { // Daca pixelul este negru
                    for (int k = 0; k < 8; k++) {
                        int ni = i + dx[k];
                        int nj = j + dy[k];
                        if (IsInside(source, ni, nj) && aux.at<uchar>(ni, nj) == 255) {
                            dst.at<uchar>(i, j) = 255; // Dilatare: pixelul devine alb daca are un vecin alb
                            break;
                        }
                    }
                }
            }
        }
    }
    return dst;
}

Mat create_red_mask(image_channels_hsv hsv_channels) {
    Mat mask = Mat::zeros(hsv_channels.H.size(), CV_8UC1);

    for (int i = 0; i < hsv_channels.H.rows; i++) {
        for (int j = 0; j < hsv_channels.H.cols; j++) {
            float hue = hsv_channels.H.at<float>(i, j);
            float sat = hsv_channels.S.at<float>(i, j) * 255;
            float val = hsv_channels.V.at<float>(i, j) * 255;

            if ((hue <= 5 || hue >= 200) && sat > 80 && val > 70) {
                mask.at<uchar>(i, j) = 255;
            }
        }
    }

    Mat erodedMask = erosion(mask, 1);
    Mat finalMask = dilation(erodedMask, 1);

    return finalMask;
}

Mat color_labels(labels labels_str){

    /*
     * This method will generate a number of no_labels colors and
     * generate a color image containing each label displayed in a different color
     */

    int rows = labels_str.labels.rows;
    int cols = labels_str.labels.cols;
    int no_labels = labels_str.no_labels;
    Mat result = Mat(rows, cols, CV_8UC3);
    Vec3b* colors = new Vec3b[no_labels + 1];

    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < no_labels; i++) {
        colors[i] = Vec3b(rand()%255, rand()%255, rand()%255);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int l = labels_str.labels.at<int>(i,j);
            result.at<Vec3b>(i, j) = colors[l];
        }
    }
    //*****END OF YOUR CODE(DO NOT DELETE / MODIFY THIS LINE) *****

    return result;
}

labels Two_pass_labeling(Mat source) {
    Mat labels;
    int rows, cols, no_newlabels;

    /*
     * This method will implement the two pass labeling algorithm
     * Hint:
     *  Use the vector structure from C++(actually you need a vector of vectors and a simple one check out the lab works)
     *  You can use queue from C++ with its specific actions (push, pop, empty, front)
     */

    rows = source.rows;
    cols = source.cols;
    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    labels = Mat(rows, cols, CV_32SC1, Scalar(0));
    int label = 0;

    vector<vector<int>> edges(1000);

    int np_di[4] = { -1, 0, 0, 1 };
    int np_dj[4] = { 0, -1, 1, 0 };

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (source.at<uchar>(i, j) == 255 && labels.at<int>(i, j) == 0) {
                vector<int> L;

                for (int k = 0; k < 4; k++) {
                    int ni = i + np_di[k];
                    int nj = j + np_dj[k];
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && labels.at<int>(ni, nj) > 0) {
                        L.push_back(labels.at<int>(ni, nj));
                    }
                }

                if (L.empty()) {
                    label++;
                    labels.at<int>(i, j) = label;
                } else {
                    int x = *min_element(L.begin(), L.end());
                    labels.at<int>(i, j) = x;

                    for (int y : L) {
                        if (y != x) {
                            edges[x].push_back(y);
                            edges[y].push_back(x);
                        }
                    }
                }
            }
        }
    }

    vector<int> newlabels(label + 1, 0);
    int newlabel = 0;

    for (int i = 1; i <= label; i++) {
        if (newlabels[i] == 0) {
            newlabel++;
            queue<int> Q;
            newlabels[i] = newlabel;
            Q.push(i);

            while (!Q.empty()) {
                int x = Q.front();
                Q.pop();

                for (int y : edges[x]) {
                    if (newlabels[y] == 0) {
                        newlabels[y] = newlabel;
                        Q.push(y);
                    }
                }
            }
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (labels.at<int>(i, j) > 0) {
                labels.at<int>(i, j) = newlabels[labels.at<int>(i, j)];
            }
        }
    }

    no_newlabels = newlabel;

    //*****END OF YOUR CODE(DO NOT DELETE / MODIFY THIS LINE) *****

    return {labels, no_newlabels};
}

float compute_thinness_ratio(int area, int perimeter) {
    /*
     * This method will compute the thinness ratio and will return it
     */

    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    float thinness_ratio = 4 * PI * (static_cast<float>(area) / static_cast<float>(perimeter * perimeter));
    //*****END OF YOUR CODE(DO NOT DELETE / MODIFY THIS LINE) ****

    return thinness_ratio;
}

Mat get_object_instance(Mat source, Vec3b color) {
    /*
     * This method will save in a different matrix in a binary format(0 and 255)
     * the selected object
     */
    Mat result = Mat::zeros(source.size(), CV_8UC1);
    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
            if (source.at<Vec3b>(i, j) == color) {
                result.at<uchar>(i, j) = 255;
            } else {
                result.at<uchar>(i, j) = 0;
            }
        }
    }

    //*****END OF YOUR CODE(DO NOT DELETE / MODIFY THIS LINE) ****

    return result;
}

perimeter naive_perimeter(Mat binary_object) {
    /*
     * This method will compute the perimeter and save the contour in a perimeter structure
     * that will store the two components
     */

    perimeter object_perimeter;
    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    object_perimeter.contour = Mat::zeros(binary_object.size(), CV_8UC1);
    object_perimeter.length = 0;

    int y_coord_neighborhood[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    int x_coord_neighborhood[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

    for (int i = 1; i < binary_object.rows - 1; i++) {
        for (int j = 1; j < binary_object.cols - 1; j++) {
            if (binary_object.at<uchar>(i, j) == 255) {
                bool is_edge = false;
                for (int k = 0; k < 8; k++) {
                    int ni = i + y_coord_neighborhood[k];
                    int nj = j + x_coord_neighborhood[k];
                    if (binary_object.at<uchar>(ni, nj) == 0) {
                        is_edge = true;
                        break;
                    }
                }
                if (is_edge) {
                    object_perimeter.contour.at<uchar>(i, j) = 255;
                    object_perimeter.length++;
                }
            }
        }
    }

    //*****END OF YOUR CODE(DO NOT DELETE / MODIFY THIS LINE) ****

    return object_perimeter;
}

int compute_area(Mat binary_object) {
    /*
     * This method will compute the object area and return it
     */

    int area = 0;

    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for (int i = 0; i < binary_object.rows; i++) {
        for (int j = 0; j < binary_object.cols; j++) {
            if (binary_object.at<uchar>(i, j) == 255) {
                area++;
            }
        }
    }

    //*****END OF YOUR CODE(DO NOT DELETE / MODIFY THIS LINE) ****

    return area;
}

struct ObjectInfo {
    float thinness_ratio;
    int label;
    int centroid_i;
    int centroid_j;
};

Mat extract_circular_objects(Mat source) {
    labels labeling_result = Two_pass_labeling(source);
    Mat labels = labeling_result.labels;
    int no_labels = labeling_result.no_labels;

    Mat circular_objects = Mat::zeros(source.size(), CV_8UC1);
    vector<ObjectInfo> object_ratios;

    // Praguri ajustate
    const float min_thinness_ratio = 0.4;
    const float min_area_ratio = 0.0005;  // Aria minima ca proportie din imagine
    const float max_area_ratio = 0.01;    // Aria maxima ca proportie din imagine
    const int min_area_absolute = 50;     // Aria minima absoluta
    const float max_y_diff_ratio = 0.1;   // Diferența maxima pe axa Y 10% din inaltime

    for (int label = 1; label <= no_labels; label++) {
        Mat object_instance = Mat::zeros(source.size(), CV_8UC1);
        for (int i = 0; i < source.rows; i++) {
            for (int j = 0; j < source.cols; j++) {
                if (labels.at<int>(i, j) == label) {
                    object_instance.at<uchar>(i, j) = 255;
                }
            }
        }

        int area = compute_area(object_instance);
        perimeter obj_perimeter = naive_perimeter(object_instance);

        if (obj_perimeter.length > 0 && area > min_area_absolute) {
            float thinness_ratio = compute_thinness_ratio(area, obj_perimeter.length);
            float area_ratio = static_cast<float>(area) / (source.rows * source.cols);

            if (thinness_ratio >= min_thinness_ratio &&
                area_ratio >= min_area_ratio &&
                area_ratio <= max_area_ratio) {
                int sum_i = 0, sum_j = 0, count = 0;
                for (int i = 0; i < source.rows; i++) {
                    for (int j = 0; j < source.cols; j++) {
                        if (object_instance.at<uchar>(i, j) == 255) {
                            sum_i += i;
                            sum_j += j;
                            count++;
                        }
                    }
                }
                int centroid_i = count > 0 ? sum_i / count : 0;
                int centroid_j = count > 0 ? sum_j / count : 0;
                object_ratios.push_back({thinness_ratio, label, centroid_i, centroid_j});
            }
        }
    }

    // sortare dupa thinness ratio descrescator
    sort(object_ratios.begin(), object_ratios.end(),
         [](const ObjectInfo& a, const ObjectInfo& b) {
             return a.thinness_ratio > b.thinness_ratio;
         });

    // selectam maxim 2 obiecte, verifica dif pe axa Y
    vector<ObjectInfo> selected_objects;
    int max_y_diff = static_cast<int>(source.rows * max_y_diff_ratio);

    for (size_t i = 0; i < object_ratios.size() && selected_objects.size() < 2; i++) {
        bool valid = true;

        // daca exista deja un obiect selectat, verificam diferenta pe axa Y
        if (!selected_objects.empty()) {
            int y_diff = abs(selected_objects[0].centroid_i - object_ratios[i].centroid_i);
            if (y_diff > max_y_diff) {
                valid = false;
            }
        }

        if (valid) {
            selected_objects.push_back(object_ratios[i]);
        }
    }

    // marcam obiectele selectate
    for (const auto& obj : selected_objects) {
        int label = obj.label;
        for (int i = 0; i < source.rows; i++) {
            for (int j = 0; j < source.cols; j++) {
                if (labels.at<int>(i, j) == label) {
                    circular_objects.at<uchar>(i, j) = 255;
                }
            }
        }
    }

    // debugging, afisam centrul
    Mat debug_image = circular_objects.clone();
    cvtColor(debug_image, debug_image, COLOR_GRAY2BGR);
    for (const auto& obj : selected_objects) {
        circle(debug_image, Point(obj.centroid_j, obj.centroid_i), 5, Scalar(0, 255, 0), -1);
    }
    imshow("Debug: Selected Circular Objects", debug_image);

    return circular_objects;
}

Mat correct_red_eyes(Mat original_image, Mat circular_objects) {
    Mat corrected_image = original_image.clone();

    for (int i = 0; i < circular_objects.rows; i++) {
        for (int j = 0; j < circular_objects.cols; j++) {
            if (circular_objects.at<uchar>(i, j) == 255) {
                Vec3b &pixel = corrected_image.at<Vec3b>(i, j);
                uchar green = pixel[1];
                uchar blue = pixel[0];
                int average = (blue + green) / 2;
                pixel[2] = average;
                pixel[1] = average;
                pixel[0] = average;
            }
        }
    }

    return corrected_image;
}