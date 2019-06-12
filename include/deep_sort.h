#ifndef DEEP_SORT_H_
#define DEEP_SORT_H_

#include <string>
#include <map>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "tracker.h"

using namespace std;

struct Args{
    float min_confidence;
    int min_detection_height;
    float nms_max_overlap;
    float max_nn_distance;
    float max_iou_distance;
    int nn_budget;
    bool display;
    Args()
    {
        min_confidence = 0.6;
        min_detection_height = 0;
        nms_max_overlap = 1.0;
        max_nn_distance = 0.2;
        max_iou_distance = 0.7;
        nn_budget = 100;
        display = false;
    }
};


class DeepSort
{
    public:
        DeepSort(bool display=false, float min_confidence=0.6, 
                float min_detection_height=0, float nms_max_overlap=1.0, 
                int nn_budget=100, float max_nn_distance=0.2, 
                float max_iou_distance=0.7);
        ~DeepSort();
        vector<vector<float> > track(vector<float> , vector<vector<float> >);

    private:
        cv::Scalar create_unique_color(int tag);
        void draw_tracks(cv::Mat* , vector<Track> , float, vector<float>);
        void draw_detections(cv::Mat* , vector<Detection> , float, vector<float>);
        vector<Detection> create_detection(vector<vector<float> >, float, float , int);

    private:
        Tracker* tracker_;
        Args args_;

};
#endif
