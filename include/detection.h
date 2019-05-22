#ifndef DETECTION_H
#define DETECTION_H

#include <vector>
#include <cstdlib>

using namespace std;

class Detection
{
    public:
        Detection(vector<float> tlwh, float confidence, vector<float> feature);
        vector<float> to_tlbr();
        vector<float> to_xyah();

        vector<float> tlwh_;
        float confidence_;

    public:
        vector<float> feature_;
};

#endif
