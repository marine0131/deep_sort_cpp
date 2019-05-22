#include "detection.h"
/*
     * a bouding box detection in current frame
     *
     * Parameters
     * tlwh: vector
     *     Bouding box in format '[x, y, w, h]'
     *  confidence: vector
     *     detector confidence
     *  feature: vector
     *     the features of the object
     *
     *  Attributes
     * tlwh: vector
     *     Bouding box in format '[x, y, w, h]'
     *  confidence: vector
     *     detector confidence
     *  feature: vector
     *     the features of the object
     */
Detection::Detection(vector<float> tlwh, float confidence, vector<float> feature)
{
    tlwh_ = tlwh;
    confidence_ = confidence;
    feature_ = feature;
}

vector<float> Detection::to_tlbr()
{
/*
 * convert bounding box to format '(min x, min y, max x, max y)'
 */
    vector<float> tblr;

    tblr.push_back(tlwh_[0]);
    tblr.push_back(tlwh_[1]);
    tblr.push_back(tlwh_[0]+tlwh_[2]);
    tblr.push_back(tlwh_[1]+tlwh_[3]);

    return tblr;
}

vector<float> Detection::to_xyah()
{
    /*
     * convert bounding box to format (center_x, center_y, aspect ratio, height), aspect ratio = width/height
     */
    vector<float> xyah;
    xyah.push_back(tlwh_[0]+ tlwh_[2]/2.0);
    xyah.push_back(tlwh_[1]+ tlwh_[3]/2.0);
    xyah.push_back(tlwh_[2]/tlwh_[3]);
    xyah.push_back(tlwh_[3]);
    return xyah;
}
