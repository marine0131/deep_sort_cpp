#include "matching_metric.h"
#include <iostream>
#include <cmath>
/*
 * distance metric, for each target, returns the closest distance to any sample that has been observed so far
 * Parameters:
 * --------
 *  budget: optional [int] default=-1
 *      if not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached
 *
 *  Attributes
 *  -------
 *  samples: Dict [int -> list[ndarray]]
 *      dict that maps from target identities to the list of samples that have been observed so far
 *
 */

DistanceMetric::DistanceMetric(int budget)
{
    budget_ = budget;
}

DistanceMetric::~DistanceMetric(){};

Eigen::MatrixXf DistanceMetric::distance_metric(string metric_name, 
        vector<Track> tracks, vector<Detection> detections, 
        vector<int> track_indices, vector<int> detection_indices)
{
    /* use different cost method accrding to the metric name
     *
     */
    Eigen::MatrixXf cost_matrix;
    if(metric_name == "cosine"|| metric_name == "euclidean")
        cost_matrix = nn_cost(metric_name, &samples_, tracks, detections, track_indices, detection_indices);
    else if(metric_name == "iou")
        cost_matrix = iou_cost(tracks, detections, track_indices, detection_indices);
    else
    {
        cerr << "Invalid metric, must be either 'euclidean' or 'cosine'" << endl;
        exit(-1);
    }

    return cost_matrix;
}

void DistanceMetric::partial_fit(vector<vector<vector<float> > > features, vector<int> targets)
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
        if(it != samples_.end())
        {
            
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

