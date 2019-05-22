#include "track.h"
#include <iostream>

/*
 * single track with with state space (x, y)
 * (x,y) is planar coordinate
 * Parameters
 * mean: vector
 *     mean vecotr of the initial state distribution
 * cov: vector
 *     covariance matrix of the initial state distribution
 * track_id: int
 *     a unique track identifier
 * n_init: int
 *     number of consecutive detections debfore the track is confirmed
 * max_age: int
 *     max number of consecutive misses before the track state is set to 'Deleted'
 * feature: vector (optional)
 *     feature vector of the detection this track originates from
 *
 * Attributes:
 * mean: vector
 *     mean vecotr of the initial state distribution
 * cov: vector
 *     covariance matrix of the initial state distribution
 * track_id: int
 *     a unique track identifier
 * hits: int
 *     total number of measurment paired with this track
 * age: int
 *     total number of frames since first occurance
 * time_since_update: int
 *     total number of frames since last measurement update
 * state: TrackState
 *     current track state
 * features: 2d vector
 *     cache of features
 */
Track::Track(Eigen::Matrix<float, 8, 1> mean, Eigen::Matrix<float,8,8>cov, int track_id, int n_init, int max_age, vector<float> feature)
{
    mean_ = mean;
    cov_ = cov;
    track_id_ = track_id;
    n_init_ = n_init;
    max_age_ = max_age;

    hits_ = 1;
    age_ = 1;
    time_since_update_ = 0;
    state_ = Tentative;

    if(feature.size() != 0)
        features_.push_back(feature);
}

vector<float> Track::to_tlwh()
{
    /*
     * get current position in bounding box in format 'top left x, top left y, width, height
     */

    vector<float> ret(4);
    ret = vector<float>(mean_.data(), mean_.data()+4);
    // cout << "ret: " << ret[0] << "," << ret[1] << "," << ret[2] << "," << ret[3] << endl;
    ret[2] *= ret[3];
    ret[0] -= ret[2]/2.0;
    ret[1] -= ret[3]/2.0;
    return ret;
}

vector<float> Track::to_tlbr()
{
    /*
     * get current position in bounding box in format 'top left x, top left y, bottom right x, bottom right y
     */

    vector<float> ret(4);
    ret = to_tlwh();
    ret[2] = ret[0] + ret[2];
    ret[3] = ret[1] + ret[3];
    return ret;
}

void Track::predict(KalmanFilter* kf)
{
    /*
     * propagate the state distribution to the current time step 
     * using kalman filter 
     */

    kf->predict(&mean_, &cov_);
    age_ ++;
    time_since_update_ ++;
}

void Track::update(KalmanFilter* kf, Detection detection)
{
    /*
     * detection paired with this track
     * preform kf's measurement update step adn update feature cache
     */
    vector<float> temp_measurement;
    temp_measurement = detection.to_xyah();
    Eigen::Vector4f measurement (temp_measurement.data());
    kf->update(&mean_, &cov_, measurement);
    features_.push_back(detection.feature_);

    hits_ ++;
    time_since_update_ = 0;
    if(state_ == Tentative && hits_ >= n_init_)
        state_ = Confirmed;
}

void Track::mark_missed()
{
    /* mark this track missed
     */
    if(state_ == Tentative)
        state_ = Deleted;
    else if(state_ == Confirmed && time_since_update_ > max_age_)
        state_ = Deleted;
}

bool Track::is_tentative()
{
    return state_ == Tentative;
}

bool Track::is_confirmed()
{
    return state_ == Confirmed;
}

bool Track::is_deleted()
{
    return state_ == Deleted;
}

// int main(int argc, char** argv)
// {
//     KalmanFilter kf;
//     Eigen::Vector4d measurement (0,0,1,1);
//     Eigen::Matrix<float,8,1> mean = Eigen::Matrix<float, 8, 1>::Zero();
//     Eigen::Matrix<float,8,8> cov = Eigen::Matrix<float, 8, 8>::Zero();
//     kf.initiate(measurement, &mean, &cov);
// 
//     for(size_t i = 0; i < cov.cols(); i++)
//         cov(i, i) = 1e-5;
//     vector<float> feature = vector<float>(6,4);
//     Track track(mean, cov, 1, 10, 30, feature);
//     track.predict(kf);
//     track.update(kf, );
//     cout <<track.mean_<<endl;
//     cout << track.features_ <<endl;
// }
