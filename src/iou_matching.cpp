#include <algorithm>
#include <iostream>
#include "iou_matching.h"


vector<float> iou(vector<float> bbox, vector<vector<float> > candidates)
{
    /*
     *  computer intersection over union
     *  Parameters
     *  ------------
     *  bbox: vector
     *     a bounding box in format '(top left x, top left y, width, height)'
     *  candidates: vector<vector float>
     *      matrix of candidate bounding boxes(one per row) in the same format 
     *      as 'bbox'
     *
     *  Returns
     *  --------
     *  vector<float>
     *      the intersection over union in [0, 1] between the 'bbox' and each
     *      candidate. 
     *      use 1.0 - the intersection over union
     *      a lower score means a larger fraction of the 'bbox' is
     *      occluded by the candidate. means more seperated
     */

    // vector<vector<float> > tl, br, wh;
    // vector<float> area_candidates, area_intersection;

    float area_bbox = bbox[2] * bbox[3];
    vector<float> iou_ratio;

    for(vector<vector<float> >::iterator it = candidates.begin(); it != candidates.end(); it ++)
    {
        vector<float> tl_, br_, wh_;
        float area_intersection, area_candidates;
        tl_.push_back(max(bbox[0], (*it)[0]));
        tl_.push_back(max(bbox[1], (*it)[1]));
        // tl.push_back(tl_);
        br_.push_back(min(bbox[0]+bbox[2], (*it)[0]+(*it)[2]));
        br_.push_back(min(bbox[1]+bbox[3], (*it)[1]+(*it)[3]));
        // br.push_back(br_);
        wh_.push_back(max(0.0f, br_[0]-tl_[0]));
        wh_.push_back(max(0.0f, br_[1]-tl_[1]));
        // wh.push_back(wh_);
        area_intersection = wh_[0]*wh_[1];
        area_candidates = (*it)[2] * (*it)[3];

        iou_ratio.push_back(1.0f - area_intersection/(area_bbox+area_candidates-area_intersection));
    }

    return iou_ratio;
}

Eigen::MatrixXf iou_cost(vector<Track> tracks, vector<Detection> detections, vector<int> track_indices, vector<int> detection_indices)
{
    /*  cost matrix calculated by iou
     *
     *  Parameters
     *  ---------
     *  tracks: [Track]
     *      vector of tracks
     *  detections: [Detection]
     *      vector of detections
     *  track_indices: Optional [int]
     *      vector of indices to tracks that be matched, defaults to all 'tracks'
     *  detection_indices: Optional [int]
     *      vector of indices to detections that be matched, defaults to all 
     *      'detections'
     *
     *  Returns
     *  ---------
     *  cost_matrix: vector<vector<float> >
     *      cost matrix of shape len(track_indices)*len(detection_indices) where
     *      entry [i,j] is '1-iou(tracks[track_indices[i]], detections[
     *      detection_indices[j]])'
     */

    // vector<vector<float> > cost_matrix(track_indices.size(), vector<float>(detection_indices.size(), 0));
    Eigen::MatrixXf cost_matrix;

    cost_matrix.resize(track_indices.size(), detection_indices.size());

    // set candidates
    vector<vector<float> > candidates;
    for(vector<int>::iterator it = detection_indices.begin(); it != detection_indices.end(); it++)
    {
        candidates.push_back(detections[*it].tlwh_);
    }
    // cout << "candidates: ";
    // for(vector<vector<float> >::iterator it=candidates.begin(); it!=candidates.end(); ++it)
    //     cout << (*it)[0] << "," << (*it)[1] << "," << (*it)[2] << "," << (*it)[3] << endl;

    // set track bbox
    for(size_t row = 0; row < track_indices.size(); row++)
    {
        int track_idx = track_indices[row];
        if(tracks[track_idx].time_since_update_ > 1)
        {
            for(size_t i = 0; i < detection_indices.size(); i++)
                cost_matrix(row, i) = INFTY_COST;
            continue;
        }

        vector<float> bbox;
        bbox = tracks[track_idx].to_tlwh();


        // cout << "bbox: ";
        // cout << bbox[0] << "," << bbox[1] << "," << bbox[2] << "," << bbox[3] << endl;
        vector<float> cost_vector = iou(bbox, candidates);
        cost_matrix.row(row) = Eigen::VectorXf::Map(&cost_vector[0], cost_vector.size());
    }
    return cost_matrix;
}


/*
 * test iou function
 */
// int main(int argc, char** argv)
// {
//     vector<float> iou_ratio;
//     vector<float> bbox = {1,1,1,1};
//     vector<vector<float> > candidates; 
//     vector<float> candi = {1.1, 1.1, 1,1};
//     candidates.push_back(candi);
//     for(size_t i =0; i<5;i++)
//     {
//         candi[0] += 0.1;
//         candi[1] += 0.1;
//         candidates.push_back(candi);
//     }
//     iou_ratio = iou(bbox, candidates);
//     for(vector<float>::iterator it = iou_ratio.begin(); it != iou_ratio.end(); it++)
//         cout << *it << endl;
// }
