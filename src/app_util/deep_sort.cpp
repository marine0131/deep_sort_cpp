#include <algorithm>
#include "nms.hpp"
#include "deep_sort.h"


DeepSort::DeepSort(bool display, float min_confidence, float min_detection_height, 
        float nms_max_overlap, int nn_budget, float max_nn_distance, 
        float max_iou_distance)
{
    args_.display = display;
    args_.min_confidence = min_confidence;
    args_.min_detection_height = min_detection_height;
    args_.nms_max_overlap = nms_max_overlap;
    args_.nn_budget = nn_budget;
    args_.max_nn_distance = max_nn_distance;
    args_.max_iou_distance = max_iou_distance;

    int max_age = 30;
    int n_init = 3;
    // creat a tracker
    tracker_ = new Tracker("cosine", args_.max_nn_distance, args_.max_iou_distance, 
            max_age, n_init, args_.nn_budget);
}

DeepSort::~DeepSort()
{
   delete tracker_;
}


cv::Scalar DeepSort::create_unique_color(int tag)
{
    /*
     * use a tag as seed to create a unique color
     *
     * returns
     * --------
     *  color Scalar
     */
    float hue_step = 0.61;
    float h = (tag * hue_step) - (int)(tag*hue_step);
    float v = 1.0f - (int(tag * hue_step) % 4) / 5.0f;           
    float s = 1.0f;

    int i = int(h*6.0);
    float f = (h*6.0) - i;
    float p = v*(1.0 - s);
    float q = v*(1.0 - s*f);
    float t = v*(1.0 - s*(1.0-f));
    int vint = (int)(v*255);
    int tint = (int)(t*255);
    int pint = (int)(p*255);
    int qint = (int)(q*255);
    i = i%6;
    switch(i)
    {
        case 0:
            return cv::Scalar(vint, tint, pint);
            break;
        case 1:
            return cv::Scalar(qint, vint, pint);
            break;
        case 2:
            return cv::Scalar(pint, vint, tint);
            break;
        case 3:
            return cv::Scalar(pint, qint, vint);
            break;
        case 4:
            return cv::Scalar(tint, pint, vint);
            break;
        case 5:
            return cv::Scalar(vint, pint, qint);
            break;
        default:
            return cv::Scalar(vint, vint, vint);
            break;
    }
}

void DeepSort::draw_tracks(cv::Mat* im, vector<Track> tracks, float zoom, 
        vector<float> translation)
{
    /*
     * draw tracks on self defined image each track has unique color and unique id
     * Parameters:
     * ---------
     *  im: image
     *  tracks: vector<Track>
     *      current tracks
     *  zoom: float
     *  translation : [x, y]
     *      these two params used for transformation from real world frame to 
     *      image frame
     */
    for(vector<Track>::iterator it=tracks.begin(); it!= tracks.end(); ++it)
    {
        if(!it->is_confirmed()||it->time_since_update_>0)
            continue;
        cv::Scalar color = create_unique_color(it->track_id_);
        
        //self.viewer.rectangle(*track.to_tlwh().astype(np.int), label=str(track.track_id))
        cv::Point pt1, pt2;
        vector<float> tlwh = it->to_tlwh();
        pt1.x = (int)(tlwh[0]*zoom + translation[0]);
        pt1.y = (int)(tlwh[1]*zoom + translation[1]);
        pt2.x = (int)((tlwh[0] + tlwh[2])*zoom + translation[0]);
        pt2.y = (int)((tlwh[1] + tlwh[3])*zoom + translation[1]);
        cv::rectangle(*im, pt1, pt2, color, 2);

        string label = to_string(it->track_id_);
        int baseline=1;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 1, 2, &baseline);
        cv::Point center;
        center.x = pt1.x + 5;
        center.y = pt1.y + 5 + text_size.height;                  
        pt2.x = pt1.x + 10 + text_size.width;
        pt2.y = pt1.y + 10 + text_size.height;
        cv::rectangle(*im, pt1, pt2, color, -1);
        cv::putText(*im, label, center, cv::FONT_HERSHEY_PLAIN,1, cv::Scalar(255, 255, 255), 2);
    }
}

void DeepSort::draw_detections(cv::Mat* im, vector<Detection> detections, float zoom, 
        vector<float> translation)
{
    /*
     * draw detections on self defined image detections are all white color
     * Parameters:
     * ---------
     *  im: image
     *  detections: vector<detection>
     *      current detections
     *  zoom: float
     *  translation : [x, y]
     *      these two params used for transformation from real world frame to 
     *      image frame
     */
    for(vector<Detection>::iterator it = detections.begin(); it != detections.end(); ++it)
    {
        cv::Point pt1, pt2;
        pt1.x = (int)(it->tlwh_[0]*zoom + translation[0]);
        pt1.y = (int)(it->tlwh_[1]*zoom + translation[1]);
        pt2.x = (int)((it->tlwh_[0]+it->tlwh_[2])*zoom + translation[0]);
        pt2.y = (int)((it->tlwh_[1]+it->tlwh_[3])*zoom + translation[1]);

        cout << "draw detection: (" << pt1.x<<","<<pt1.y<<") ("<< pt2.x << ", " << pt2.y<< ")" << endl;

        cv::rectangle(*im, pt1, pt2, cv::Scalar(255, 255, 255), 2);
    }
}

vector<Detection> DeepSort::create_detection(vector<vector<float> > detection_mat, 
        float min_height, float min_confidence, int feature_dim)
{
    /*
     * Create detections
     * Parameters:
     * -----------
     *  detection_mat: float mat
     *      each row is one detection [id x_min y_min width height confidence 
     *      [features]]
     *  min_height: float
     *      a deteciton with height lower than this value will be discarded.
     *  min_confidence: float
     *      a detection with confidence lower than this value will be discaded.
     *  feature_dim: int
     *      features dimension
     * Returns:
     * -----------
     *  vector<Detection>
     */
    vector<Detection> detections;
    for(vector<vector<float> >::iterator it = detection_mat.begin(); it != detection_mat.end(); ++it)
    {
        vector<float> bbox(4);
        copy(it->begin()+1, it->begin()+5, bbox.begin());
        if(bbox[3] < min_height)
            continue;

        float confidence = (*it)[5];
        if(confidence < min_confidence)
            continue;

        vector<float> feature(feature_dim);
        copy(it->begin()+6, it->end(), feature.begin());

        detections.push_back(Detection(bbox, confidence, feature));
    }
    return detections;
}


vector<vector<float> > DeepSort::track(vector<float> tlbr, vector<vector<float> > detections_mat)
{
    /*
     * run the track
     * Parameters:
     * ---------------
     *  tlbr: [x_min, y_min, x_max, y_max]
     *  detections_mat: detections in matrix
     *      [id, x_min, y_min, width, height, confidence, [features] ]
     */
    clock_t init_time=clock();
    cout << endl << "Start processing ................................" << endl;
    // generate detections
    vector<Detection> detections = create_detection(detections_mat, 
            args_.min_detection_height, args_.min_confidence, detections_mat[0].size()-6);

    // run non-maxima suppression
    vector<vector<float> > boxes;
    vector<float> scores;
    for(vector<Detection>::iterator it = detections.begin(); 
            it!=detections.end(); ++it)
    {
        boxes.push_back(it->to_tlbr());
        scores.push_back(it->confidence_);
    }
    vector<int> pick = nms(boxes, args_.nms_max_overlap, scores);
    vector<Detection> nms_detections;
    for(vector<int>::iterator it = pick.begin(); it!=pick.end();++it)
    {
        nms_detections.push_back(detections[*it]);
    }
    detections.clear();
    detections = nms_detections;
    cout << "detections: "<< detections.size() <<endl;

    // predict tracker
    clock_t startTime = clock();
    tracker_->predict();
    cout << "predict time: " <<  (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;
    startTime = clock();

    // update tracker
    tracker_->update(detections);
    cout << "update time: " <<  (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;

    if(args_.display)
    {
        // read blank image 
        cv::Mat im(600, 800, CV_8UC3, cv::Scalar(0,0,0));

        //draw detections
        float zoom_x = 800.0 / (1.2*(tlbr[2]-tlbr[0]));
        float zoom_y = 600.0 / (1.2*(tlbr[3]-tlbr[1]));
        float zoom = zoom_x>zoom_y?zoom_y:zoom_x;
        vector<float> translation(2,0.0);
        translation[0] = 400 - (tlbr[2]+tlbr[0])/2.0*zoom;
        translation[1] = 300 - (tlbr[3]+tlbr[1])/2.0*zoom;
        draw_detections(&im, detections, zoom, translation);

        //draw tracks
        draw_tracks(&im, tracker_->tracks_, zoom, translation);

        cv::imshow("image", im);

        int wait_time = 100 - (float)(clock()-init_time)*1000/CLOCKS_PER_SEC;
        wait_time = wait_time>1?wait_time:1;
        cv::waitKey(wait_time);
    }
    nms_detections.clear();
    detections.clear();

    vector<vector<float> > track_box;
    for(vector<Track>::iterator it=tracker_->tracks_.begin(); it!=tracker_->tracks_.end(); ++it)
    {
        if(!it->is_confirmed()||it->time_since_update_>0)
            continue;
        vector<float> tmp;
        tmp.push_back(it->track_id_);
        vector<float> t = it->to_tlwh();
        for(vector<float>::iterator i=t.begin(); i!=t.end(); ++i)
            tmp.push_back(*i);
        track_box.push_back(tmp);
    }

    return track_box;
}



int main(int argc, char** argv)
{
    // args
    // min_confidence = 0.6;
    // min_detection_height = 0;
    // nms_max_overlap = 0.8;
    // nn_budget = 50;
    // max_cosine_distance = 0.2;
    // display = true;
    DeepSort deep_sort(true);
    vector<float> tlbr = {0.0, 0.0, 1.5, 2.4};
    vector<vector<float> > detections;
    vector<float> detection = {126, 1.106338, 1.921463, 0.421436, 0.282616, 1.00, 0.070090, 0.138561, 0.123765, 0.1, 57170, 1328.965576, 2.105685, 986.691162, 0.704052};
    detections.push_back(detection);

    deep_sort.track(tlbr, detections);
}
