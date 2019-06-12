#ifndef DEEP_SORT_APP_H
#define DEEP_SORT_APP_H 

#include <string>
#include <map>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "tracker.h"
using namespace std;

struct Args{
    string sequence_dir;
    string detection_file;
    string output_file;
    float min_confidence;
    int min_detection_height;
    float nms_max_overlap;
    float max_cosine_distance;
    float max_iou_distance;
    int nn_budget;
    bool display;
    Args()
    {
        sequence_dir = "";
        detection_file = "";
        output_file = "/tmp/hypotheses.txt";
        min_confidence = 0.8;
        min_detection_height = 0;
        nms_max_overlap = 1.0;
        max_cosine_distance = 0.2;
        max_iou_distance = 0.7;
        nn_budget = 10000;
        display = true;
    }
};

struct SeqInfo{
    string sequence_name;
    map<int, string> image_filenames;
    vector<vector<float> > detections;
    string groundtruth;
    vector<int> image_size;
    int min_frame_idx;
    int max_frame_idx;
    int feature_dim;
    float update_ms;
};

cv::Scalar create_unique_color(float tag);
vector<Detection> create_detection(vector<vector<float> > detection_mat, int frame_idx, int min_height, float min_confidence);
void gather_sequence_info(string sequence_dir, string detection_file);
void run(Args args);
#endif
