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
    clock_t startTime = clock();

    if(!data_is_normalized)
    {
        // Eigen::MatrixXf a2, b2;
        // a2 = a.array().square();
        // b2 = b.array().square();
        //Eigen::MatrixXf na, nb;
        // na = a.rowwise().norm();//sum().cwiseSqrt();
        // nb = b.rowwise().norm();//sum().cwiseSqrt();
        // cout << "cosine_time: " << (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;
        // for(size_t i = 0; i < a.rows(); i++)
        //     for(size_t j = 0 ; j < a.cols(); j ++)
        //         a(i, j) = a(i, j)/ na(i);
        // for(size_t i = 0; i < b.rows(); i++)
        //     for(size_t j = 0 ; j < b.cols(); j ++)
        //         b(i, j) = b(i, j)/ nb(i);
        a.rowwise().normalize();
        b.rowwise().normalize();
    }
    cout << "cosine_time: " << (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;
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
    distances = cosine_distance(a, b);

    c.resize(distances.cols(),1);
    for(size_t i = 0; i < distances.cols(); i ++)
        c(i) = distances.col(i).minCoeff();
    return c;
}



/*
 * nearest neighbo distance metric, for each target, returns the closest distance to any sample that has been observed so far
 * Parameters:
 * --------
 *  metric: string
 *      "euclidean" or "cosine"
 *  matching_threshold: float
 *      samples with larger distance are considered an invalid match
 *  budget: optional [int] default=-1
 *      if not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached
 *
 *  Attributes
 *  -------
 *  samples: Dict [int -> list[ndarray]]
 *      dict that maps from target identities to the list of samples that have been observed so far
 *
 */

NearestNeighborDistanceMetric::NearestNeighborDistanceMetric(string metric, float matching_threshold, int budget)
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

    matching_threshold_ = matching_threshold;
    budget_ = budget;
}

NearestNeighborDistanceMetric::~NearestNeighborDistanceMetric(){};

void NearestNeighborDistanceMetric::partial_fit(vector<vector<float> > features, vector<int> targets, vector<int> active_targets)
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
    map<int, Eigen::MatrixXf>::iterator it;
    for(size_t i = 0; i < targets.size(); i++)
    {
        it = samples_.find(targets[i]);
        if(it != samples_.end())
        {
            // find target index in already exist samples
            int ind = targets[i];
            it->second.conservativeResize(it->second.rows()+1, Eigen::NoChange);
            it->second.row(it->second.rows()-1) = Eigen::VectorXf::Map(
                    &features[i][0], features[i].size());
            if(budget_ > 0 && budget_ < it->second.rows())
            {
                it->second = it->second.bottomRows(budget_);
            }
        }
        else
        {
            Eigen::MatrixXf temp_mat = Eigen::VectorXf::Map(&features[i][0], features[i].size());
            samples_.insert(pair<int, Eigen::MatrixXf>(targets[i],temp_mat.transpose())); 
        }
    }

    map<int, Eigen::MatrixXf> new_samples;
    for(size_t i = 0; i < active_targets.size(); i++)
        new_samples[active_targets[i]] = samples_[active_targets[i]];

    samples_ = new_samples;
}

Eigen::MatrixXf NearestNeighborDistanceMetric::distance(Eigen::MatrixXf features, vector<int> targets)
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
    // cout << "enter NearestNeighborDistanceMetric distance...." << endl;
    Eigen::MatrixXf cost_matrix;

    int L = features.rows();
    int T = targets.size();
    cost_matrix.resize(T, L); 

    for(size_t i = 0; i < T; i ++)
    {
        Eigen::MatrixXf  c;
        c = metric_(samples_.find(targets[i])->second, features); //samples[target[i]] N*M, features L*M, return c is L length
        cost_matrix.row(i) = c.transpose();
    }
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
//     NearestNeighborDistanceMetric nn("euclidean", 1, 2);
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
