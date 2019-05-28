#include<iostream>
#include <algorithm>
#include <unordered_set>
#include "tracker.h"
#include "iou_matching.h"
#include <time.h>

// NNDistanceMetric * DistanceMetric;

/*
 * Parameters:
 * metric: nn_matching/NNDistanceMetric
 *     a distance metric for measurement-to-track association
 * max_age: int
 *     Maximum number of missed misses before a track is deletec
 * n_init: int
 *     number of consecutive detections before the track is confirmed.
 *     the track state is set to 'deleted' if a miss occures within the first 
 *     'n_init' frame.
 * 
 * Atttributes:
 * metric: nn_matching/NNDistanceMetric
 *     a distance metric for measurement-to-track association
 * max_age: int
 *     Maximum number of missed misses before a track is deletec
 * n_init: int
 *     number of consecutive detections before the track is confirmed.
 *     the track state is set to 'deleted' if a miss occures within the first 
 *     'n_init' frame.
 * kf: kalman_filter/KalmanFilter
 *     a kalman filter to filter target trajectories
 * tracks: [Track]
 *     
 */
Tracker::Tracker(string metric, float max_nn_distance, float max_iou_distance, int max_age, int n_init, int nn_budget)
{
    max_iou_distance_ = max_iou_distance;
    max_nn_distance_ = max_nn_distance;
    max_age_ = max_age;
    n_init_ = n_init;
    next_id_ = 1;
    kf_ = new KalmanFilter();

    distance_metric_ = new NNDistanceMetric(metric, nn_budget);
}

Tracker::~Tracker(){}

void Tracker::predict()
{
    /*
     * propagate track state distributions one time step forward
     * predict should be called once every time step, before 'update'
     */
    for(vector<Track>::iterator it = tracks_.begin(); it != tracks_.end(); ++it)
    {
        it->predict(kf_);
    }
}

void Tracker::update(vector<Detection> detections)
{
    /*
     * Perform measurement update and track managment
     * Param// enters
     * -------
     *  detections: vector<Detection>
     *      a list of detections at the current time step
     */

    clock_t startTime = clock();
    // run matching cascade
    vector<Match> matches;
    vector<int> unmatched_tracks, unmatched_detections;
    match_(detections, &matches, &unmatched_tracks, &unmatched_detections);
    cout << "match_time: " << (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;
    startTime = clock();

    // for(vector<Track>::iterator it = tracks_.begin(); it != tracks_.end(); it ++)
    //     cout << "track_id(before update): " << it->track_id_ << ",time_since_update_: " << it->time_since_update_ <<endl;
    // cout <<"next_id_ : " << next_id_<<endl;

    // update track set
    for(vector<Match>::iterator it = matches.begin(); it != matches.end(); it++)
        tracks_[it->track_idx].update(kf_, detections[it->detection_idx]);

    for(vector<int>::iterator it = unmatched_tracks.begin(); it != unmatched_tracks.end(); it ++)
        tracks_[*it].mark_missed();
    for(vector<int>::iterator it = unmatched_detections.begin(); it != unmatched_detections.end(); it ++)
        initiate_track_(detections[*it]);

    // for(vector<Track>::iterator it = tracks_.begin(); it != tracks_.end(); it ++)
    //     cout << "track_id(after update): " << it->track_id_ << ",time_since_update_: " << it->time_since_update_ <<endl;

    // delete tracks that state is is_deleted ans extract tracks that state is confirmed
    vector<int> targets;
    vector<vector<vector<float> > >features;
    vector<vector<float> > feat;

    for(vector<Track>::iterator it = tracks_.begin(); it != tracks_.end();)
    {
        if(it->is_deleted())
            it = tracks_.erase(it);
        else
        {
            if(it->is_confirmed())
            {
                // tmp_feat.resize(it->features_.size(), it->features_[0].size());
                // for(size_t i = 0; i < it->features_.size(); ++i)
                //{
                    // tmp_feat.row(i) = Eigen::VectorXf::Map(
                    //        &(it->features_[i][0]), it->features_[i].size());
                //}
                targets.push_back(it->track_id_);
                features.push_back(it->features_);
                it->features_.clear();
                //features.push_back(tmp_feat);
            }
            it ++;
        }
    }

    cout << "update_time: " << (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;
    startTime = clock();
    // update distance metric
    distance_metric_->partial_fit(features, targets);
    cout << "fit_time: " << (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;
    
}


void Tracker::match_(vector<Detection> detections, vector<Match>* matches, 
        vector<int>* unmatched_tracks, vector<int>* unmatched_detections)
{
    /* match method
     * Paramters
     * -----------
     *  detections: 
     *
     *
     */
    // split track set into confirmed and unconfirmed tracks
    vector<int> confirmed_tracks, unconfirmed_tracks, detection_indices;
    for(size_t i = 0; i < tracks_.size(); ++ i)
    {
        if(tracks_[i].is_confirmed())
            confirmed_tracks.push_back(i);
        else
            unconfirmed_tracks.push_back(i);
    }
    for(size_t i = 0; i < detections.size(); ++ i)
    {
        detection_indices.push_back(i);
    }

    // for(vector<Detection>::iterator it = detections.begin(); it < detections.end(); ++it)
    // {
    //     vector<float>tlwh = it->tlwh_;
    //     for(vector<float>::iterator iit = tlwh.begin(); iit!= tlwh.end(); ++iit)
    //         cout << "\t" << *iit << ",";
    //     cout << endl;
    // }
    // cout << "confirmed_tracks: " << confirmed_tracks.size() << "; unconfirmed_tracks: " << unconfirmed_tracks.size() << endl;
    cout << "now tracks: [";
    for(vector<Track>::iterator it = tracks_.begin(); it != tracks_.end(); ++it)
        cout << it->track_id_ << ", ";
    cout << "]"<<endl;

    // associate confirmed tracks using appearance features
    vector<Match> matches_a, matches_b;
    vector<int> unmatched_tracks_a, unmatched_tracks_b;
    vector<int> unmatched_detections_a;

    clock_t startTime = clock();

    matching_cascade(distance_metric_, nn_cost, max_nn_distance_, max_age_, 
            tracks_, detections, &matches_a, &unmatched_tracks_a, 
            &unmatched_detections_a, confirmed_tracks, detection_indices);

    cout << "cost_match_time: " << (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;
    startTime = clock();
    // cout << "matches_a: ";
    // for(vector<Match>::iterator it=matches_a.begin(); it != matches_a.end(); ++it)
    //     cout << "(" << it->track_idx << "," << it->detection_idx << ")" ;
    // cout << endl;
    // cout << "unmatched_tracks_a: [" ;
    // for(vector<int>::iterator it=unmatched_tracks_a.begin(); it != unmatched_tracks_a.end(); ++it)
    //     cout << *it << ",";
    // cout << "] " << endl;
    // cout << "unmatched_detections_a: [" ;
    // for(vector<int>::iterator it=unmatched_detections_a.begin(); it != unmatched_detections_a.end(); ++it)
    //     cout << *it << ",";
    // cout << "] " << endl;
    // associate remaining tracks together with unconfirmed tracks using IOU
    vector<int> iou_track_candidates=unconfirmed_tracks;
    for(vector<int>::iterator it=unmatched_tracks_a.begin(); it != unmatched_tracks_a.end();) 
    {
        if(tracks_[*it].time_since_update_ == 1)
        {
            iou_track_candidates.push_back(*it);
            it = unmatched_tracks_a.erase(it);
        }
        else
            it++;
    }
    // cout << "iou_track_candidates: " << iou_track_candidates.size() << endl;
    // cout << "unmatched_tracks_a(erased): " << unmatched_tracks_a.size() << endl;

    min_cost_matching(distance_metric_, iou_cost, max_iou_distance_, tracks_, 
            detections, &matches_b, &unmatched_tracks_b, unmatched_detections, 
            iou_track_candidates, unmatched_detections_a);

    cout << "iou_match_time: " << (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;
    startTime = clock();
    // cout << "matches_b: [";
    // for(vector<Match>::iterator it=matches_b.begin(); it != matches_b.end(); ++it)
    //     cout << "(" << it->track_idx << "," << it->detection_idx << ")" ;
    // cout << "]" << endl;
    // cout << "unmatched_tracks_b: [" ;
    // for(vector<int>::iterator it=unmatched_tracks_b.begin(); it != unmatched_tracks_b.end(); ++it)
    //     cout << *it << ",";
    // cout << "] " << endl;
    // cout << "unmatched_detections_b: [" ;
    // for(vector<int>::iterator it=unmatched_detections->begin(); it != unmatched_detections->end(); ++it)
    //     cout << *it << ",";
    // cout << "] " << endl;

    *matches = matches_a;
    for(vector<Match>::iterator it=matches_b.begin(); it!=matches_b.end(); it ++)
        matches->push_back(*it);
    for(vector<int>::iterator it=unmatched_tracks_b.begin(); it != unmatched_tracks_b.end(); it ++)
        unmatched_tracks_a.push_back(*it);
    unordered_set<int> unmatched_tracks_set;
    copy(unmatched_tracks_a.begin(), unmatched_tracks_a.end(), inserter(unmatched_tracks_set, unmatched_tracks_set.end()));
    copy(unmatched_tracks_set.begin(), unmatched_tracks_set.end(), back_inserter(*unmatched_tracks));

    // cout << "final matches: ";
    // for(vector<Match>::iterator it=matches->begin(); it != matches->end(); ++it)
    //     cout << "(" << it->track_idx << "," << it->detection_idx << ")" ;
    // cout << "]" <<endl;
    // cout << "final unmatched_tracks: [";
    // for(vector<int>::iterator it=unmatched_tracks->begin(); it != unmatched_tracks->end(); ++it)
    //     cout << *it<< ", ";
    // cout << "]" << endl;
    // cout << "final unmatched_detections: [";
    // for(vector<int>::iterator it=unmatched_detections->begin(); it != unmatched_detections->end(); ++it)
    //     cout << *it<< ", ";
    // cout << "]" << endl;
}


void Tracker::initiate_track_(Detection detection)
{
    /*
     * initiate a track and added to tracks
     * Parameters:
     * ---------
     *  detection: Detection
     *
     */
    Eigen::Matrix<float, 8, 1> mean = Eigen::Matrix<float, 8, 1>::Zero();
    Eigen::Matrix<float, 8, 8> cov = Eigen::Matrix<float, 8,8>::Zero();
    kf_->initiate(detection.to_xyah(), &mean, &cov);
    tracks_.push_back(Track(mean, cov, next_id_, n_init_, max_age_, detection.feature_));
    // cout << "new track: " << next_id_ << "size: " << detection.feature_.size()<< endl;
    next_id_ ++;
}
