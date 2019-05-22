#ifndef TRACK_H
#define TRACK_H

#include <cstdlib>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "kalman_filter.h"
#include "detection.h"

using namespace std;

enum TrackState
{
    /*
     * enum for track state, newly created tracks are 'tentative', they are 
     * classified as 'confirmed' when enough evidence get, tracks no longer 
     * alive are classidied as 'deleted', and will be removed
     */
    Tentative = 1,
    Confirmed = 2,
    Deleted = 3
};

class Track
{
    public:
        Track(Eigen::Matrix<float,8,1> mean, Eigen::Matrix<float,8,8> cov, int track_id, int n_init, int max_age, vector<float> feature);

        vector<float> to_tlwh();

        vector<float> to_tlbr();

        void predict(KalmanFilter* kf);

        void update(KalmanFilter* kf, Detection detection);

        void mark_missed();

        bool is_tentative();

        bool is_confirmed();

        bool is_deleted();

    private:
        int n_init_;
        int max_age_;

    public:
        Eigen::Matrix<float, 8, 1> mean_;
        Eigen::Matrix<float,8,8> cov_;
        int track_id_;
        int hits_;
        int age_;
        int time_since_update_;
        TrackState state_;
        vector<vector<float>> features_;
};

#endif
