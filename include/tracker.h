#include <string>
#include "linear_assignment.h"


// Eigen::MatrixXf gated_metric_(vector<Track> tracks, vector<Detection> detections, vector<int> track_indices, vector<int> detection_indices);


class Tracker
{
    public:
        Tracker(string metric, float max_nn_distace, float max_iou_distance, int max_age, int n_init, int nn_buget);
        ~Tracker();
        void predict();
        void update(vector<Detection> detections);
        void initiate_track_(Detection detection);
        vector<Track> tracks_;

    private:
        void match_(vector<Detection> detections, vector<Match>* matches, vector<int>* unmatched_tracks, vector<int>* unmatched_detections);
        DistanceMetric* distance_metric_;

    private:
        float max_nn_distance_;
        float max_iou_distance_;
        int max_age_;
        int n_init_;
        KalmanFilter *kf_;
        int next_id_;

};



