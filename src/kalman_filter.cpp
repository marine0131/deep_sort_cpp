#include <iostream>
#include "kalman_filter.h"
#include <time.h>

map<int, float> chi2inv95 = {
    {1, 3.8415},
    {2, 5.9915},
    {3, 7.8147},
    {4, 9.4877},
    {5, 11.070},
    {6, 12.592},
    {7, 14.067},
    {8, 15.507},
    {9, 16.919}
};

/*
 * Kalman filter
 * a simple kalmanfilter for tracking bounding boxes in planar space
 * 8-dimensional state space
 *   x,y,a,h,vx,vy,va,vh
 * contains the bounding box and its velocity
 */
KalmanFilter::KalmanFilter()
{
    int dt = 1;
    motion_mat_ = Eigen::Matrix<float, 8, 8>::Zero();
    update_mat_ = Eigen::Matrix<float, 4, 8>::Zero();
    // create kalman filter model
    for(size_t i = 0, cols = motion_mat_.cols()/2; i < cols; i ++)
    {
        motion_mat_(i, i) = 1;
        motion_mat_(i+4, i+4) = 1;
        motion_mat_(i, i+4) = 1;
    }
    
    for(size_t i = 0, cols = update_mat_.cols()/2; i < cols; i ++)
    {
        update_mat_(i, i) = 1;
    }

    std_weight_position_ = 1.0/20.0;
    std_weight_velocity_ = 1.0/160.0;

    // cout << "\nmotion_mat: \n" << motion_mat_<< endl;
    // cout << "\nupdate_mat: \n" << update_mat_<< endl;
}

KalmanFilter::~KalmanFilter(){};

void KalmanFilter::initiate(vector<float> measurement, Eigen::Matrix<float, 8, 1>* mean, Eigen::Matrix<float, 8, 8>* cov)
{
    /*
     * create track from unassociated measurement
     *
     * Parameters
     * measurement: 4 dimentional vector
     *     bounding box coordinates (x, y, a , h) with center postion (x, y)
     *     and aspect ratio a, height h.
     *
     * Returns:
     * mean: 8 dimensional vector
     *     mean vector
     * cov: 8*8 dimensional matrix
     *     covariance matrix
     */

    for(size_t i = 0; i< measurement.size(); i ++)
        (*mean)(i) = measurement[i];

    Eigen::VectorXf std(8);
    std <<
        2*std_weight_position_ * measurement[3],
        2*std_weight_position_ * measurement[3],
        1e-2,
        2*std_weight_position_ * measurement[3],
        10*std_weight_velocity_* measurement[3],
        10*std_weight_velocity_* measurement[3],
        1e-5,
        10*std_weight_velocity_* measurement[3];

    for(size_t i = 0; i < std.size(); i ++)
        (*cov)(i, i) = std(i)*std(i);
}


void KalmanFilter::predict(Eigen::Matrix<float, 8, 1>* mean, Eigen::Matrix<float, 8, 8>* cov)
{
    /*
     * kalman prediction step
     * Parameters:
     * mean: 8*1 vector
     *   mean value of object state at previos time step
     * cov: 8*8 matrix
     *   covariance of the object state at the previos time step
     *
     * Returns:
     * mean: 8*1 vector
     *   updated mean
     * cov: 8*8 matrix
     *   updated covariance
     */

    Eigen::VectorXf std(8);
    std <<
        std_weight_position_* (*mean)(3),
        std_weight_position_* (*mean)(3),
        1e-2,
        std_weight_position_* (*mean)(3),
        std_weight_velocity_* (*mean)(3),
        std_weight_velocity_* (*mean)(3),
        1e-5,
        std_weight_velocity_* (*mean)(3);

    Eigen::Matrix<float, 8, 8> motion_cov = Eigen::Matrix<float, 8, 8>::Zero();
    for(size_t i = 0; i < std.size(); i++)
    {
        motion_cov(i, i) = std(i)*std(i);
    }

    (*mean) = motion_mat_ * (*mean);

    (*cov) = (motion_mat_ * (*cov) * (motion_mat_.transpose())) + motion_cov;
}

void KalmanFilter::project(Eigen::Matrix<float, 8, 1> mean, Eigen::Matrix<float, 8, 8> cov, Eigen::Matrix<float, 4, 1>* projected_mean, Eigen::Matrix<float, 4, 4>* projected_cov)
{
    /*
     * project state distribution to measurement space
     * Parameters:
     * mean: 8*1 vector
     *   state's mean vector
     * covariance: 8*8 vector
     *   state's covariance matrix
     * 
     * Returns:
     * updated measurement mean vector and updated measurenment covariance
     */

    Eigen::Vector4f std;
    std <<
        std_weight_position_* mean(3),
        std_weight_position_* mean(3),
        1e-1,
        std_weight_position_* mean(3);
    Eigen::Matrix4f innovation_cov = Eigen::Matrix4f::Zero();
    for(size_t i = 0; i < std.size(); i++)
        innovation_cov(i,i) = std(i)*std(i);

    (*projected_mean) = update_mat_ * mean;
    (*projected_cov) = update_mat_ * cov * update_mat_.transpose() + innovation_cov;
}

void KalmanFilter::update(Eigen::Matrix<float, 8, 1>* mean, Eigen::Matrix<float, 8, 8>* cov, Eigen::Vector4f measurement)
{
    /*
     * kalman filter update step
     * Parameters
     * mean: 8*1 vector
     *     the predicted state's mean vector
     * cov: 8*8 matrix
     *     the state's covariance matrix
     * measuremtn: 4*1 vector
     *     the measurement
     *
     * Returns
     * mean: 8*1 vector
     *     updated mean
     * cov: 8*8 matrix
     *     updated matrix
     */
    Eigen::Matrix<float, 4,1> projected_mean = Eigen::Matrix<float, 4, 1>::Zero();
    Eigen::Matrix<float, 4,4> projected_cov = Eigen::Matrix<float, 4, 4>::Zero();
    // project to measurment domain
    project(*mean, *cov, &projected_mean, &projected_cov);
    
    Eigen::Matrix<float,8,4> kalman_gain = Eigen::Matrix<float, 8, 4>::Zero();
    Eigen::Matrix<float,4,4> projected_cov_inv = Eigen::Matrix<float, 4, 4>::Zero(); 
    clock_t start = clock();
    projected_cov_inv = projected_cov.inverse();
    kalman_gain = (*cov) * update_mat_.transpose() * projected_cov_inv;

    (*mean) = (*mean) + kalman_gain * (measurement - projected_mean);
    (*cov) = (*cov) - kalman_gain * projected_cov * kalman_gain.transpose();
}

Eigen::VectorXf KalmanFilter::gating_distance(Eigen::Matrix<float, 8, 1> mean, Eigen::Matrix<float, 8,8> cov, Eigen::MatrixXf measurements, bool only_position)
{
    /*
     * computing gating distance between state distribution and measurements
     *
     * Parameters:
     * mean: 8*1
     *    mean vector of the state
     * covariance: 8*8
     *    covariance of the state distribution 
     * measurement: 4*1
     *    N*4 matrix of N measurements
     * only_position: optional[bool]
     *    if true, distance computation is done with respect to the bounding box center position only
     *
     * Returns:
     * distance: N*1
     *     i-th element contains Mahalanobis distance between (mean, covariance) and measrement[i]
     */

    Eigen::Matrix<float, 4,1> projected_mean = Eigen::Matrix<float, 4, 1>::Zero();
    Eigen::Matrix<float, 4,4> projected_cov = Eigen::Matrix<float, 4, 4>::Zero();
    project(mean, cov, &projected_mean, &projected_cov);
    if(only_position)
    {


    }

    // cholesky 
    Eigen::LLT<Eigen::Matrix4f> cholesky_mat;
    Eigen::Matrix4f sqrt_cov = projected_cov.array().abs().sqrt();
    cholesky_mat = sqrt_cov.llt();

    // solve linear equation, calculate mahalanobis distance
    int N = measurements.rows();

    Eigen::MatrixXf z;
    z.resize(N, 4);
    for(size_t i = 0; i < N; i++)
        z.row(i) = measurements.row(i)-projected_mean.transpose();

    Eigen::MatrixXf mahalanobis = cholesky_mat.solve(z.transpose());


    Eigen::VectorXf distance = (mahalanobis.array().square()).colwise().sum();
    // for(size_t i = 0; i < N; i++)
    // {
    //     Eigen::VectorXf mahalanobis = cholesky_mat.solve(measurements.row(i).transpose() - projected_mean);
    //     cout << mahalanobis << endl;
    //     distance(i) = mahalanobis.transpose() * mahalanobis;
    // }
    return distance;

}

// int main(int argc, char** argv)
// {
//     KalmanFilter kf;
//     Eigen::Vector4f measurement (0,0,1,1);
//     Eigen::Matrix<float,8,1> mean = Eigen::Matrix<float, 8, 1>::Zero();
//     Eigen::Matrix<float,8,8> cov = Eigen::Matrix<float, 8, 8>::Zero();
// 
// 
//     kf.initiate(measurement, &mean, &cov);
//     kf.predict(&mean, &cov);
//     cout << "\nmean: \n" << mean << endl;
//     cout << "\ncov: \n" << cov << endl;
// 
//     Eigen::Matrix<float,4,1> projected_mean= Eigen::Matrix<float, 4, 1>::Zero();
//     Eigen::Matrix<float,4,4> projected_cov= Eigen::Matrix<float, 4, 4>::Zero();
//     //kf.project(&mean, &cov, &projected_mean, &projected_cov);
//     //cout << "\nprojected_mean: \n" << projected_mean << endl;
//     //cout << "\nprojected_cov: \n" << projected_cov << endl;
//     kf.update(&mean, &cov, measurement);
// 
//     measurement = Eigen::Vector4f (2,2,1.2,1);
//     kf.predict(&mean, &cov);
//     kf.update(&mean, &cov, measurement);
//     cout << "\nnew_mean: \n" << mean << endl;
//     cout << "\nnew_cov: \n" << cov << endl;
// 
//     //get distance
//     Eigen::Matrix<float, 4, -1> measurements;
//     measurements.resize(4,3);
//     Eigen::Matrix<float, 3,4> a;
//     a <<
//         1,1, 1, 1,
//         1.5,1.5,1.5,1.5,
//         3,3,3,3;
//     measurements = a.transpose();
// 
//     cout << "measurements:" << measurements << endl;
//     Eigen::VectorXf distance;
//     distance.resize(measurements.cols());
//     kf.gating_distance(&mean, &cov, &measurements, &distance);
//     cout << "\ndistance: \n" << distance << endl;
// }
