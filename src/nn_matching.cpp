#include "nn_matching.h"
#include <iostream>
#include <cmath>

static Eigen::MatrixXf pdist(Eigen::MatrixXf a, Eigen::MatrixXf b)
{
    /*
     * compute pair-wise squard distance between points in 'a' and 'b'
     *
     * Parameters
     * ------
     *  a: matrix  N*M
     *      N samples of dimensionality M
     *  b: matrix L*M
     *      L samples of dimensionality M
     *
     *  Returns
     *  ----
     *  c: matrix 
     *      returns a matrix of N*L, who's element(i,j) contains the squared distance 
     *      between a.row(i) and b.row(j)
     */

    Eigen::MatrixXf c;

    if(a.rows() == 0 || b.rows() == 0)
        return c;
    int N = a.rows();
    int L = b.rows();
    int M = a.cols();

    Eigen::MatrixXf square_a, square_b;
    square_a.resize(N, 1);
    square_b.resize(L, 1);
    for(size_t i = 0; i < N; i ++)
        for(size_t j = 0; j < M; j ++)
            square_a(i) += a(i, j)*a(i,j);

    for(size_t i = 0; i < L; i ++)
        for(size_t j = 0; j < M; j ++)
            square_b(i) += b(i, j)*b(i,j);
    
    c.resize(N, L);
    c = -2 * a * (b.transpose());
    for(size_t i = 0; i < N; i ++)
        for(size_t j = 0; j < L; j ++)
            c(i, j) += square_a(i) + square_b(j);

    return c.cwiseMax(0);
}

static Eigen::MatrixXf cosine_distance(Eigen::MatrixXf a, Eigen::MatrixXf b, bool data_is_normalized)
{
    /*
     * compute paire-wise cosine distance between points in 'a' and 'b'
     * Parameters:
     * -------
     *  a: matrix  N*M
     *      N samples of dimensionality M
     *  b: matrix L*M
     *      L samples of dimensionality M
     *  data_is_normalized: bool
     *      if True, assumes rows in a and b are unit length vectors
     *      otherwise, a and b are explicitly normalized to length 1
     *
     *  Returns
     *  ----
     *  c: matrix 
     *      returns a matrix of N*L, who's element(i,j) contains the squared distance 
     *      between a.row(i) and b.row(j)
     */

    if(!data_is_normalized)
    {
        a.rowwise().normalize();
        b.rowwise().normalize();
    }
    return  1.0 - (a * b.transpose()).array();
}

static Eigen::MatrixXf nn_euclidean_distance(Eigen::MatrixXf a, Eigen::MatrixXf b)
{
    /*
     * compute paire-wise cosine distance between points in 'a' and 'b'
     * Parameters:
     * -------
     *  a: matrix  N*M
     *      N samples of dimensionality M
     *  b: matrix L*M
     *      L samples of dimensionality M
     *  data_is_normalized: bool
     *      if True, assumes rows in a and b are unit length vectors
     *      otherwise, a and b are explicitly normalized to length 1
     *
     *  Returns
     *  ----
     *  c: vector 
     *      returns a L length vector, each element contains the smallest euclidean distance of a sample in b and all samples in a
     */

    Eigen::MatrixXf c;

    //calculate distance of each samples in a and b
    Eigen::MatrixXf distances;
    distances = pdist(a, b);

    //find b vs each a's smallest distance
    c.resize(distances.cols(),1);
    for(size_t i = 0; i < distances.cols(); i ++)
        c(i) = max(0.0f, distances.col(i).minCoeff());

    return c;
}


static Eigen::MatrixXf nn_cosine_distance(Eigen::MatrixXf a, Eigen::MatrixXf b)
{
    /*
     * compute paire-wise cosine distance between points in 'a' and 'b'
     * Parameters:
     * -------
     *  a: matrix  N*M
     *      N samples of dimensionality M
     *  b: matrix L*M
     *      L samples of dimensionality M
     *  data_is_normalized: bool
     *      if True, assumes rows in a and b are unit length vectors
     *      otherwise, a and b are explicitly normalized to length 1
     *
     *  Returns
     *  ----
     *  c: vector 
     *      returns a L length vector, each element contains the smallest euclidean distance of a sample in b and all samples in a
     */

    Eigen::MatrixXf distances, c;
    distances = cosine_distance(a, b, false);

    c.resize(distances.cols(),1);
    for(size_t i = 0; i < distances.cols(); i ++)
    {
        float min = distances.col(i).minCoeff();
        c(i) = min; //>0?min:0.0f;
    }
    return c;
}



/*
 * nearest neighbo distance metric, for each target, returns the closest distance to any sample that has been observed so far
 * Parameters:
 * --------
 *  metric: string
 *      "euclidean" or "cosine"
 *  budget: optional [int] default=-1
 *      if not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached
 *
 *  Attributes
 *  -------
 *  samples: Dict [int -> list[ndarray]]
 *      dict that maps from target identities to the list of samples that have been observed so far
 *
 */

NNDistanceMetric::NNDistanceMetric(string metric, int budget)
{
    if(metric == "euclidean")
        metric_ = nn_euclidean_distance;
    else if(metric == "cosine")
        metric_ = nn_cosine_distance;
    else
    {
        cerr << "Invalid metric, must be either 'euclidean' or 'cosine'" << endl;
        exit(-1);
    }

    budget_ = budget;
}

NNDistanceMetric::~NNDistanceMetric(){};

void NNDistanceMetric::partial_fit(vector<vector<vector<float> > > features, vector<int> targets)
{
    /*
     * update the distance metric with new data
     * Parameters
     * ------
     *  features: N*M matrix
     *      N features with dimensionality M, they are the features of targets
     *  targets: N vector
     *      An integer array of associated target identities
     *  active_targets: vector
     *      vector of targets that are currentlu present in the scene
     */
    //map<int, Eigen::MatrixXf>::iterator it;
    map<int, vector<vector<float> > >::iterator it;
    for(size_t i = 0; i < targets.size(); i++)
    {
        it = samples_.find(targets[i]);
        // int feat_row = features[i].rows();
        // int feat_col = features[i].cols();
        if(it != samples_.end())
        {
            // find target index in already exist samples
            // it->second.conservativeResize(it->second.rows()+feat_row, Eigen::NoChange);
            // it->second.block(it->second.rows()-feat_row, 0, feat_row, feat_col) = features[i];
            // if(budget_ > 0 && budget_ < it->second.rows())
            //     it->second = it->second.bottomRows(budget_);
            
            for(vector<vector<float> >::iterator iit= features[i].begin(); iit!=features[i].end(); ++iit)
                it->second.push_back(*iit);
            if(budget_ > 0 && budget_ < it->second.size())
            {
                for(size_t j = 0;j < it->second.size()-budget_; ++j)
                {
                    it->second.erase(it->second.begin());
                }
            }

        }
        else
        {
            // it is a new target
            // samples_.insert(pair<int, Eigen::MatrixXf>(targets[i],features[i])); 
            samples_.insert(pair<int, vector<vector<float> > >(targets[i],features[i])); 
        }
    }

    for(it = samples_.begin(); it != samples_.end(); )
    {
        if(find(targets.begin(), targets.end(), it->first)==targets.end())
            it = samples_.erase(it);
        else
            it++;
    }
}

Eigen::MatrixXf NNDistanceMetric::distance(Eigen::MatrixXf features, 
        vector<int> targets)
{
    /*
     * compute distance features and targets
     *
     * Parameters
     * --------
     *  features: L*M matrix
     *      L features of dimensionality M
     *  targets: T vector of int
     *      list of targets identities to match 'features'
     *
     *  Returns:
     *  -------
     *  cost_matrix: T * L matrix
     *      element(i,j) contains the closest squared distance between 'targets[i]'(already exist target) and 'features[j]'(current detected target)
     */
    // cout << "enter NNDistanceMetric distance...." << endl;
    Eigen::MatrixXf cost_matrix;

    clock_t startTime = clock();

    int L = features.rows();
    int T = targets.size();
    cost_matrix.resize(T, L); 

    for(size_t i = 0; i < T; i ++)
    {
        Eigen::MatrixXf  c;
        Eigen::MatrixXf target;
        vector<vector<float> > mat = samples_.find(targets[i])->second;
        target.resize(mat.size(), mat[0].size());
        for(size_t i = 0; i < mat.size(); i++)
            target.row(i)=Eigen::VectorXf::Map(&mat[i][0], mat[i].size());

        c = metric_(target, features); //samples[target[i]] N*M, features L*M, return c is L length
        cost_matrix.row(i) = c.transpose();
    }
    cout <<": nn_match_time: " << (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;
    return cost_matrix;
}


Eigen::MatrixXf NNDistanceMetric::gate_cost_matrix(KalmanFilter kf, 
        Eigen::MatrixXf cost_matrix, vector<Track> tracks, 
        vector<Detection> detections, vector<int> track_indices, 
        vector<int> detection_indices, float gated_cost, bool only_position)
{
    /*
     * invalidate infeasible entries in cost matrix based on the state distribution
     * obtained by kalman filter
     *
     * Parameters:
     * ----------
     *  kf: KalmanFilter
     *  cost_matrix: MatrixXf
     *      the N*M dimensional cost matrix, where N is the bumber of track indices
     *      and M is the number of detection indices
     *  tracks: vector<Track>
     *      vector of Track at current time step
     *  detections: vector<Detection>
     *      vector of detections at current time step
     *  track_indices: vector<int>
     *      vector of track indices that maps rows in 'cost_matrix' to tracks in
     *      'tracks'
     *  detection_indices: vector<int>
     *      vector of detection indices that maps cols in 'cost_matrix' to detections
     *      int 'detections'
     *  gated_cost: Optional[float]
     *      Entries in the cost matrix corresponding to indeasible association are
     *      set this value. defaults to a very large value
     *  only_position: Optional[bool]
     *      if true, only x, y of state distribution is considered, Default to false
     *
     *  Returns:
     *  -------
     *  cost_matrix:
     *      modified cost matrix
     */

    int gating_dim = 4;
    if(only_position)
        gating_dim = 2;

    float gating_threshold = chi2inv95[gating_dim];

    Eigen::MatrixXf measurements;
    measurements.resize(detection_indices.size(), 4);
    for(size_t i = 0; i < detection_indices.size(); ++i)
    {
        vector<float> tmp = detections[detection_indices[i]].to_xyah();
        measurements.row(i) = Eigen::VectorXf::Map(&tmp[0], tmp.size());
    }

    Eigen::VectorXf gating_distance_;
    for(size_t i = 0; i < track_indices.size(); ++i)
    {
        Track track = tracks[track_indices[i]];
        gating_distance_ = kf.gating_distance(track.mean_, track.cov_, measurements, only_position);

        for(size_t j = 0; j < gating_distance_.size(); j++)
            if(gating_distance_(j) > gating_threshold) 
                cost_matrix(i, j) = gated_cost;
    }
    return cost_matrix;
}



Eigen::MatrixXf nn_cost(NNDistanceMetric* distance_metric, vector<Track> tracks, vector<Detection> detections, vector<int> track_indices, vector<int> detection_indices)
{
    // cout << "enter gated_metric_ ...." << endl;
    Eigen::MatrixXf features;
    vector<int> targets;
    features.resize(detection_indices.size(), detections[0].feature_.size());

    for(size_t i = 0; i < detection_indices.size(); ++i)
    {
        int detection_idx = detection_indices[i];
        features.row(i) = Eigen::VectorXf::Map(&(detections[detection_idx].feature_[0]), detections[detection_idx].feature_.size());
    }
    for(size_t i = 0; i < track_indices.size(); ++i)
        targets.push_back(tracks[track_indices[i]].track_id_);

    //get cost matrix
    Eigen::MatrixXf cost_matrix = distance_metric->distance(features, targets);

    //modify cost matrix
    KalmanFilter kf;
    cost_matrix = distance_metric->gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices);
    return cost_matrix;
}

// int main(int argc, char** argv)
// {
//     Eigen::MatrixXf a, b, c;
//     a.resize(2,3);
//     b.resize(4,3);
//     a <<
//         1,1,1,
//         2,2,2;
//     b <<
//         3,3,3,
//         4,4,4,
//         5,5,5,
//         6,6,6;
// 
// 
//     // pdist(a, b, &c);
//     // cosine_distance(a, b, &c);
//     // nn_euclidean_distance(a, b, &c);
//     // nn_cosine_distance(a, b, &c);
//     // cout << c << endl;
//     //
//     
// 
//     // partial_fit.........
//     NNDistanceMetric nn("euclidean", 1, 2);
//     Eigen::MatrixXf features;
//     features.resize(3,4);
//     features <<
//         1,2,3,4,
//         3,4,5,6,
//         5,6,7,8;
//     vector<int> targets = {101,102,103};
//     vector<int> active_targets = {101,102};
//     nn.partial_fit(features, targets, active_targets);
//     for (map<int,Eigen::MatrixXf>::iterator it=nn.samples_.begin(); it != nn.samples_.end(); ++it)
//         cout << it->second << endl;
//         cout << "\n\n"<< endl;
// 
//     features <<
//         1,1,1,1,
//         2,2,2,2,
//         3,3,3,3;
//     vector<int> targets1 = {101,102,103};
//     vector<int> active_targets1 = {101,102,103};
//     nn.partial_fit(features, targets1, active_targets1);
//     for (map<int,Eigen::MatrixXf>::iterator it=nn.samples_.begin(); it != nn.samples_.end(); ++it)
//         cout << it->second << "\n"<< endl;
// 
// 
//     // distance..........
//     features.resize(2,4);
//     features <<
//         7,7,2,4,
//         2,1,5,6;
//     vector<int> targets2 = {101,102,103};
//     Eigen::MatrixXf cost_matrix;
//     nn.distance(features, targets2, &cost_matrix);
//     cout << cost_matrix << endl;
//     
// }
