#include "linear_assignment.h"
#include "hungarian_alg.h"
#include <iostream>

void min_cost_matching(NNDistanceMetric* metric, Metric distance_metric, 
        float max_distance, vector<Track> tracks, vector<Detection> detections, 
        vector<Match>* matches, vector<int>* unmatched_tracks,
        vector<int>* unmatched_detections, vector<int> track_indices,
        vector<int> detection_indices)
{
    /*
     * solve linear assignment peoblem
     * Parameters
     * -------------
     *  distance_metric: callable Eigen::MatrixXf (vector<Track>, vector<Detection>
     *      vector<int>, vector<int>
     *      the distance metric is given a list of tracks and detections as well
     *      as a list of N track indices and M detection indices. the metric
     *      should return N*M dimensional cost_matrix
     *  max_distance: float
     *      gating threshold. associations with cost larger than this value are
     *      disregarded
     *  tracks: list[Tracks]
     *  detections: list[Detection]
     *  track_indices: list[int]
     *      the indices of tracks used for calculating cost_matrix, default is 
     *      [], which means all of the tracks
     *  detection_indices: list[int]
     *      the indices of detections used for calculating cost_matrix, default is 
     *      [], which means all of the detections
     *
     * returns
     * ---------
     *  matches: vector<Match>
     *      matched indices of tracks and detections
     *  unmatched_tracks: vector<int>
     *      list of unmatched track indices
     *  unmatched_detections: vector<int>
     *      list of unmatched detection indices
     */
    // cout << "enter min_cost_matching......" << endl;
    if(track_indices.size()<1 || detection_indices.size()<1)
    {
        *unmatched_tracks = track_indices;
        *unmatched_detections = detection_indices;
        return;
    }

    Eigen::MatrixXf cost_matrix;
    cost_matrix = distance_metric(metric, tracks, detections, track_indices, detection_indices);
    cost_matrix = cost_matrix.array().min(max_distance+1e-5);


    // use hungarian alg solve linear assignment problem
    AssignmentProblemSolver aps;
    size_t N = cost_matrix.rows();
    size_t M = cost_matrix.cols();
    bool transposed = false;
    Eigen::MatrixXf cost_vector;
    if(N>M)
    {
        cost_vector = cost_matrix.transpose();
        transposed = true;
        size_t tmp = M;
        M = N;
        N = tmp;
    }
    else
        cost_vector = cost_matrix;

    cost_vector.resize(1,N*M);
    const vector<float> cost_matrix_vec(cost_vector.data(), cost_vector.data()+N*M);
    vector<int> assignment(N);
    aps.Solve(cost_matrix_vec, N, M, assignment, AssignmentProblemSolver::optimal);
    // concatinete
    vector<vector<int> > indices(2, vector<int>(N, -1));
    if(transposed)
    {
        for(size_t i = 0; i < assignment.size(); ++i)
        {
            indices[1][i] = i;
            indices[0][i] = assignment[i];
        }
    }
    else
    {
        for(size_t i = 0; i < assignment.size(); ++i)
        {
            indices[0][i] = i;
            indices[1][i] = assignment[i];
        }
    }

    // for(size_t i = 0; i < indices[0].size(); ++i)
    //     cout <<"assignment: " << indices[0][i] << ", " << indices[1][i] <<endl;
    
    // clear all  return matrix
    matches->clear(); 
    unmatched_detections->clear();
    unmatched_tracks->clear();
    // process detections not in matches
    for(size_t i = 0; i < detection_indices.size(); i++)
    {
        if(find(indices[1].begin(), indices[1].end(), i) == indices[1].end())
            unmatched_detections->push_back(detection_indices[i]);
    }
    // process tracks not in matches
    for(size_t i = 0; i < track_indices.size(); i++)
    {
        if(find(indices[0].begin(), indices[0].end(), i) == indices[0].end())
            unmatched_tracks->push_back(track_indices[i]);
    }
    // process matches and tracks not in matches and matches that has large cost
    for(int i = 0; i < indices[0].size(); ++i)
    {
        int row = indices[0][i];
        int col = indices[1][i];
        int track_idx = track_indices[row];
        int detection_idx = detection_indices[col];
        if(cost_matrix(row, col) > max_distance)
        {
            unmatched_tracks->push_back(track_idx);
            unmatched_detections->push_back(detection_idx);
        }
        else
        {
            Match match= {track_idx, detection_idx};
            matches->push_back(match);
        }
    }
}

void matching_cascade(NNDistanceMetric* metric, Metric distance_metric, 
        float max_distance, int cascade_depth, vector<Track> tracks, 
        vector<Detection> detections, vector<Match>* matches, 
        vector<int>* unmatched_tracks, vector<int>* unmatched_detections, 
        vector<int> track_indices, vector<int> detection_indices)
{
    /*
     * matching existed tracks and current detections
     * Parameter
     * -------
     *  distance_metric: callable function
     *      distance metric is given a list of tracks and detections, as well as
     *      a list of N tracks indices and M detection indices. The metric should
     *      return N*M dimensional cost matrix, where element (i, j) is the
     *      association cost between i-th track in the given track indices and 
     *      j-th detection in the given detection indices
     *
     *  max_distance: float
     *      gating threshold, associations with cost larger than this value are 
     *      disregarded
     *
     *  cascade_depth: int
     *      cascade depth should be see to the maximum track age
     *  tracks: vector<Track>
     *      list of predicted tracks at the current time step
     *  detections: vector<Detection>
     *      list of detections at the current time step
     *  track_indices: Optional  vector<int>
     *      list of track indices. maps to row of cost matrix, default to all tracks
     *  detection_indices: Optional vectot<int>
     *      list of detection indices. maps to column of cost matrix, default to all detections
     *
     * Returns
     * --------
     *  matched:vector<tuple>
     *      matched track and detection indices 
     *  unmatched_tracks: vector<int>
     *      umatched track indices
     *  unmatched_detections: vector<int>
     *      unmatched detection indices
     */


    // cout << "enter matching_cascade....." << endl;
    *unmatched_detections = detection_indices;
    for(int level = 0; level < cascade_depth; level ++)
    {
        if(unmatched_detections->size() < 1) //no detection left
            break;

        vector<int> track_indices_l;
        for(vector<int>::iterator it = track_indices.begin(); 
                it != track_indices.end(); it ++)
            if(tracks[*it].time_since_update_ == 1+level)
                track_indices_l.push_back(*it);
        if(track_indices_l.size() < 1) //nothing to match ath this level
            continue;

        // for(vector<int>::iterator it = track_indices.begin(); it != track_indices.end(); it ++)
        // {
        //     cout << "track_id: " << tracks[*it].track_id_ << ",time_since_update_: " << tracks[*it].time_since_update_ <<endl;
        // }
        // cout << "level: " << level << " track_indices_l: " << track_indices_l.size() <<endl;
        vector<int> unmatched_tracks_l;
        vector<Match> matches_l;
        min_cost_matching(metric, distance_metric, max_distance, tracks, detections,
                &matches_l, &unmatched_tracks_l, unmatched_detections, 
                track_indices_l , *unmatched_detections);

        for(vector<Match>::iterator it=matches_l.begin(); it != matches_l.end(); it ++)
        {
            matches->push_back(*it);
            // delete the matched tracks in track_indices
            vector<int>::iterator iit;
            iit = find(track_indices.begin(), track_indices.end(), it->track_idx);
            if(iit != track_indices.end())
                track_indices.erase(iit);
        }
    }
    *unmatched_tracks = track_indices;

}




// int main(int argc, char** argv)
// {
//     Eigen::MatrixXf bb;
//     bb.resize(3,2);
//     bb <<
//         1,2,
//         5,4,
//         12,1;
//     // b = b.array().min(10.1);
//     // cout << b <<endl;
// 
//     AssignmentProblemSolver aps;
//     Eigen::MatrixXf b;
//     if(bb.rows() > bb.cols())
//         b = bb.transpose();
// 
//     Eigen::MatrixXf b_evec = b;
//     b_evec.resize(1,6);
//     const vector<float> b_vec(b_evec.data(), b_evec.data()+b_evec.rows()*b_evec.cols());
//     vector<int> assignment;
//     aps.Solve(b_vec, b.rows(), b.cols(), assignment, AssignmentProblemSolver::optimal);
//     for(vector<int>::iterator it=assignment.begin(); it != assignment.end(); it++)
//         cout << *it << endl;
// }
//     vector<vector<int> > a={{1,2,3,4},{1,2,3,5}};
//     vector<int> c = {1,2,3,3};
//     a.push_back(c);
//     for(size_t i = 0; i< a.size(); i++)
//     {
//         for(size_t j = 0; j < a[0].size(); j++)
//             cout << a[i][j] << ",";
//         cout <<endl;
//     }
// 
//     // for(vector<vector<float> >::iterator it=b.begin(); it != b.end(); it++)
//     // {
//     //     for(vector<float>::iterator iit=it->begin(); iit != it->end(); iit++)
//     //         cout << *iit ;
//     //     cout << endl;
//     //  }
//     // vector<int> track_indices = {1,2,3,4,5,6,7,8,9,0,10};
//     // vector<vector<int> > matches = {{1,2},{3,4},{5,6},{7,8},{9,0}};
//     // for(vector<vector<int> >::iterator it=matches.begin(); it != matches.end(); it ++)
//     // {
//     //     // delete the matched tracks in track_indices
//     //     vector<int>::iterator iit;
//     //     iit = find(track_indices.begin(), track_indices.end(), (*it)[0]);
//     //     if(iit != track_indices.end())
//     //         track_indices.erase(iit);
//     // }
//     // for(vector<int>::iterator it=track_indices.begin(); it != track_indices.end(); it++)
//     //     cout << *it << endl;
// }
