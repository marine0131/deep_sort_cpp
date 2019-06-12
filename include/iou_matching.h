#ifndef IOU_MATCHING_H
#define IOU_MATCHING_H 

#include <vector>
#include "track.h"

using namespace std;

vector<float> iou(vector<float> bbox, vector<vector<float> > candidates); 

Eigen::MatrixXf iou_cost(vector<Track> tracks, vector<Detection> detections, vector<int> track_indices, vector<int> detection_indices);

#endif
