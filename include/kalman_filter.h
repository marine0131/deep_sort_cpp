#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <Eigen/Dense>
#include <map>
#include <vector>
using namespace std;


/*
 * table for 0.95 quantile of the chi-square distribution with N degree of freedom
 * (contains values for N=1,...,9)
 *
 */

extern map<int, float> chi2inv95;
class KalmanFilter
{
    public:
        KalmanFilter();
        ~KalmanFilter();
        void initiate(vector<float> measurement, Eigen::Matrix<float, 8, 1>* mean, Eigen::Matrix<float, 8, 8>* cov);

        void predict(Eigen::Matrix<float, 8, 1>* mean, Eigen::Matrix<float, 8, 8>* cov);
        void project(Eigen::Matrix<float, 8, 1> mean, Eigen::Matrix<float, 8, 8> cov, Eigen::Matrix<float, 4, 1>* projected_mean, Eigen::Matrix<float, 4, 4>* projected_cov);
        void update(Eigen::Matrix<float, 8, 1>* mean, Eigen::Matrix<float, 8, 8>* cov, Eigen::Vector4f measurement);
        Eigen::VectorXf gating_distance(Eigen::Matrix<float, 8, 1> mean, Eigen::Matrix<float, 8,8> cov, Eigen::MatrixXf measurements, bool only_position=false);
    private:
        Eigen::Matrix<float, 8, 8> motion_mat_;
        Eigen::Matrix<float, 4, 8> update_mat_;

        float std_weight_position_;
        float std_weight_velocity_;
};

#endif
