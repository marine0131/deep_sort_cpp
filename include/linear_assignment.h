#ifndef LINEAR_ASSIGNMENT_H
#define LINEAR_ASSIGNMENT_H

#include "kalman_filter.h"
#include "track.h"
#include "detection.h"
#include "hungarian_alg.h"

#define INFTY_COST 1e5f

typedef Eigen::MatrixXf (*Metric)(vector<Track>, vector<Detection>, vector<int>,
        vector<int>);

typedef struct Match_struct
{
    int track_idx;
    int detection_idx;
}Match;

void min_cost_matching(Metric distance_metric, float max_distance, 
        vector<Track> tracks, vector<Detection> detections, 
        vector<Match>* matches, vector<int>* unmatched_tracks,
        vector<int>* unmatched_detections, 
        vector<int> track_indices,
        vector<int> detection_indices);

void matching_cascade(Metric distance_metric, float max_distance, int cascade_depth, 
        vector<Track> tracks, vector<Detection> detections, 
        vector<Match>* matches, vector<int>* unmatched_tracks, 
        vector<int>* unmatched_detections, 
        vector<int> track_indices, 
        vector<int> detection_indices);

Eigen::MatrixXf gate_cost_matrix(KalmanFilter kf, Eigen::MatrixXf cost_matrix, 
        vector<Track> tracks, vector<Detection> detections, 
        vector<int> track_indices, vector<int> detection_indices, 
        float gated_cost=INFTY_COST, bool only_position=false);


#endif
