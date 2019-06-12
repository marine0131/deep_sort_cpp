#ifndef NN_MATCHING_H 
#define NN_MATCHING_H

#include <Eigen/Core>
#include <string>
#include <map>

#include "track.h"

using namespace std;

Eigen::MatrixXf distance(string, map<int, vector<vector<float> > >*, Eigen::MatrixXf, vector<int>);

Eigen::MatrixXf gate_cost_matrix(KalmanFilter kf, Eigen::MatrixXf cost_matrix, 
        vector<Track> tracks, vector<Detection> detections, 
        vector<int> track_indices, vector<int> detection_indices, 
        float gated_cost=INFTY_COST, bool only_position=false);

Eigen::MatrixXf nn_cost(string , map<int, vector<vector<float> > >* , vector<Track>, vector<Detection>, vector<int>, vector<int>);

#endif
