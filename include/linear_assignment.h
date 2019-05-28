#ifndef LINEAR_ASSIGNMENT_H
#define LINEAR_ASSIGNMENT_H

#include "kalman_filter.h"
#include "track.h"
#include "detection.h"
#include "hungarian_alg.h"
#include "nn_matching.h"


typedef Eigen::MatrixXf (*Metric)(NNDistanceMetric*, vector<Track>, vector<Detection>, vector<int>, vector<int>);

typedef struct Match_struct
{
    int track_idx;
    int detection_idx;
}Match;

void min_cost_matching(NNDistanceMetric*, Metric distance_metric, float max_distance, 
        vector<Track> tracks, vector<Detection> detections, vector<Match>* matches, 
        vector<int>* unmatched_tracks, vector<int>* unmatched_detections, 
        vector<int> track_indices, vector<int> detection_indices);

void matching_cascade(NNDistanceMetric*, Metric distance_metric, float max_distance, 
        int cascade_depth, vector<Track> tracks, vector<Detection> detections, 
        vector<Match>* matches, vector<int>* unmatched_tracks, 
        vector<int>* unmatched_detections, vector<int> track_indices, 
        vector<int> detection_indices);



#endif
