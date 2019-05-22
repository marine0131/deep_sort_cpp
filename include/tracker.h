#include "kalman_filter.h"
#include "linear_assignment.h"
#include "iou_matching.h"
#include "nn_matching.h"
#include "track.h"
#include <string>


Eigen::MatrixXf gated_metric_(vector<Track> tracks, vector<Detection> detections, vector<int> track_indices, vector<int> detection_indices);


class Tracker
{
    public:
        Tracker(string metric, float matching_threshold, float max_iou_distance=0.7, int max_age=30, int n_init=3);
        ~Tracker();
        void predict();
        void update(vector<Detection> detections);
        void initiate_track_(Detection detection);
        vector<Track> tracks_;

    private:
        void match_(vector<Detection> detections, vector<Match>* matches, vector<int>* unmatched_tracks, vector<int>* unmatched_detections);

    private:
        float max_iou_distance_;
        int max_age_;
        int n_init_;
        KalmanFilter *kf_;
        int next_id_;

};



