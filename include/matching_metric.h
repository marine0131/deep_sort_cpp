#ifndef MATCHING_METRIC_H 
#define MATCHING_METRIC_H

#include <Eigen/Core>
#include <string>
#include <map>

#include "track.h"
#include "nn_matching.h"
#include "iou_matching.h"

using namespace std;

class DistanceMetric
{
    public:

        DistanceMetric(int budget=-1);
        ~DistanceMetric();
        Eigen::MatrixXf distance_metric(string, vector<Track>, vector<Detection>,
            vector<int> track_indices, vector<int> detection_indices);
        void partial_fit(vector<vector<vector<float> > > features, vector<int> targets);
    public:
        int budget_;
        map<int, vector<vector<float> > > samples_;
};

#endif
