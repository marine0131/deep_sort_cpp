#ifndef NN_MATCHING_H 
#define NN_MATCHING_H

#include <Eigen/Core>
#include <string>
#include <map>

#include "kalman_filter.h"
#include "track.h"

using namespace std;

static Eigen::MatrixXf pdist(Eigen::MatrixXf a, Eigen::MatrixXf b);
static Eigen::MatrixXf cosine_distance(Eigen::MatrixXf a, Eigen::MatrixXf b, bool data_is_normalized=false);
static Eigen::MatrixXf nn_euclidean_distance(Eigen::MatrixXf a, Eigen::MatrixXf b);
static Eigen::MatrixXf nn_cosine_distance(Eigen::MatrixXf a, Eigen::MatrixXf b);


class NNDistanceMetric
{
    public:

        NNDistanceMetric(string metric, int budget=-1);
        ~NNDistanceMetric();

        Eigen::MatrixXf distance(Eigen::MatrixXf features, vector<int> targets);
        void partial_fit(vector<vector<vector<float> > > features, vector<int> targets);
        Eigen::MatrixXf gate_cost_matrix(KalmanFilter kf, Eigen::MatrixXf cost_matrix, 
            vector<Track> tracks, vector<Detection> detections, 
            vector<int> track_indices, vector<int> detection_indices, 
            float gated_cost=INFTY_COST, bool only_position=false);


    private:
        Eigen::MatrixXf (*metric_)(Eigen::MatrixXf, Eigen::MatrixXf);
    public:
        int budget_;
        //map<int, Eigen::MatrixXf> samples_;
        map<int, vector<vector<float> > > samples_;
};

Eigen::MatrixXf nn_cost(NNDistanceMetric*, vector<Track>, vector<Detection>, vector<int>, vector<int>);

#endif
