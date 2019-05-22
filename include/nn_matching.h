#ifndef NN_MATCHING_H 
#define NN_MATCHING_H

#include <Eigen/Core>
#include <string>
#include <map>

using namespace std;

static Eigen::MatrixXf pdist(Eigen::MatrixXf a, Eigen::MatrixXf b);
static Eigen::MatrixXf cosine_distance(Eigen::MatrixXf a, Eigen::MatrixXf b, bool data_is_normalized=false);
static Eigen::MatrixXf nn_euclidean_distance(Eigen::MatrixXf a, Eigen::MatrixXf b);
static Eigen::MatrixXf nn_cosine_distance(Eigen::MatrixXf a, Eigen::MatrixXf b);

class NearestNeighborDistanceMetric
{
    public:

        NearestNeighborDistanceMetric(string metric, float matching_threshold, int budget=-1);
        ~NearestNeighborDistanceMetric();
        Eigen::MatrixXf distance(Eigen::MatrixXf features, vector<int> targets);
        void partial_fit(vector<vector<float> > features, vector<int> targets, vector<int> active_targets);

    private:
        Eigen::MatrixXf (*metric_)(Eigen::MatrixXf, Eigen::MatrixXf);
    public:
        int budget_;
        float matching_threshold_;
        map<int, Eigen::MatrixXf> samples_;
};

#endif
