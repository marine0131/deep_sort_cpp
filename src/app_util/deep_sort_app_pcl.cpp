#include <deep_sort_app_pcl.h>
#include "nms.hpp"
#include <fstream>
#include <algorithm>
#include <regex>

int DetectionIdx = 0;

cv::Scalar create_unique_color(int tag)
{
    float hue_step = 0.41;
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
void draw_tracks(cv::Mat* im, vector<Track> tracks, float ratio)
{
    for(vector<Track>::iterator it=tracks.begin(); it!= tracks.end(); ++it)
    {
        if(!it->is_confirmed()||it->time_since_update_>0)
            continue;
        cv::Scalar color = create_unique_color(it->track_id_);
        
        //self.viewer.rectangle(*track.to_tlwh().astype(np.int), label=str(track.track_id))
        cv::Point pt1, pt2;
        vector<float> tlwh = it->to_tlwh();
        pt1.x = (int)(tlwh[0]*ratio);
        pt1.y = (int)(tlwh[1]*ratio);
        pt2.x = (int)((tlwh[0] + tlwh[2])*ratio);
        pt2.y = (int)((tlwh[1] + tlwh[3])*ratio);
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

void draw_detections(cv::Mat* im, vector<Detection> detections, float ratio)
{
    for(vector<Detection>::iterator it = detections.begin(); it != detections.end(); ++it)
    {
        cv::Point pt1, pt2;
        pt1.x = (int)(it->tlwh_[0]*ratio);
        pt1.y = (int)(it->tlwh_[1]*ratio);
        pt2.x = (int)((it->tlwh_[0]+it->tlwh_[2])*ratio);
        pt2.y = (int)((it->tlwh_[1]+it->tlwh_[3])*ratio);

        cout << "draw detection: (" << pt1.x<<","<<pt1.y<<") ("<< pt2.x << ", " << pt2.y<< ")" << endl;

        cv::rectangle(*im, pt1, pt2, cv::Scalar(255, 255, 255), 2);
    }
}

vector<Detection> create_detection(vector<vector<float> > detection_mat, int frame_idx, int min_height, float min_confidence, int feature_dim)
{
    vector<Detection> detections;

    cout << "frame_id: "<< frame_idx << " DetectionIdx: " << DetectionIdx<< endl;
    size_t i = 0;

    for(i = DetectionIdx; i < detection_mat.size(); ++i)
    {
        if((int)detection_mat[i][0] > frame_idx)
            break;
        if((int)detection_mat[i][0] == frame_idx)
        {
            vector<float> bbox(4);
            copy(detection_mat[i].begin()+1, detection_mat[i].begin()+5, bbox.begin());
            float confidence = detection_mat[i][5];
            vector<float> feature(feature_dim);
            copy(detection_mat[i].begin()+6, detection_mat[i].end(), feature.begin());
            if(bbox[3] < min_height)
                continue;
            if(confidence < min_confidence)
                continue;

            detections.push_back(Detection(bbox, confidence, feature));
        }
    }
    DetectionIdx = i;
    return detections;
}


vector<vector<float> > read_file_to_vector(string filename)
{
    ifstream is(filename);
    if(!is)
    {
        cout << "detections file not found!"<< endl;
        exit(1);
    }
    regex pat_regex("\\d+(\\.\\d+)?");

    string line;
    vector<float> vec;
    vector<vector<float> > mat;
    while(getline(is, line))
    {
        for(sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; it != end_it; ++it)
        {
            cout << it->str() << " ";
            vec.push_back(stof(it->str()));
        }
        cout << endl;
        mat.push_back(vec);
        vec.clear();
    }

    return mat;
}

void gather_sequence_info(SeqInfo* seq_info, string sequence_dir, string detection_file)
{
    /* gather sequence information, such as image filenames, detections
     *
     * parameters
     * ---------
     *  sequence_dir: string
     *      path to MOTchallenge sequence directory
     *  detection_file: string
     *      path to the detection file
     *
     *  returns:
     *  --------
     *  info: struct
     */

    seq_info->sequence_name = sequence_dir;

    seq_info->image_file = sequence_dir+"/blank.jpg";

    if(detection_file == "")
    {
        cout << "error: need detection file, exiting..." << endl;
        exit(1);
    }

    seq_info->detections = read_file_to_vector(detection_file);

    cout << "detection shape: (" << seq_info->detections.size()  << ", "<< seq_info->detections[0].size() << ")"<< endl;

    seq_info->groundtruth = "";
    
    cv::Mat im = cv::imread(seq_info->image_file, CV_LOAD_IMAGE_COLOR);
    seq_info->image_size.push_back(im.cols);
    seq_info->image_size.push_back(im.rows);
    cout <<  "image_size: " << im.cols << ", " << im.rows << endl;

    seq_info->min_frame_idx = (*(seq_info->detections.begin()))[0];
    seq_info->max_frame_idx = (*(seq_info->detections.end()-1))[0];
     cout << "frame_idx: " << seq_info->min_frame_idx << "~" << seq_info->max_frame_idx << endl;
    
    //string info_filename = sequence_dir + "/seqInfo.ini";
    seq_info->update_ms = 1000/10.0; //hz

    seq_info->feature_dim = seq_info->detections[0].size() - 6;
    cout << "feature dim: " << seq_info->feature_dim << endl;
}


void run(Args args)
{
    SeqInfo seq_info;
    gather_sequence_info(&seq_info, args.sequence_dir, args.detection_file);

    string metric = "cosine";
    vector<float> detection_area = {12.0, 6.0};
    Tracker tracker(metric, args.max_cosine_distance, args.max_iou_distance, 30, 3, args.nn_budget);
    int frame_idx = seq_info.min_frame_idx;
    int last_idx = seq_info.max_frame_idx;
    vector<Detection> detections;
    while(frame_idx <= last_idx)
    {
        clock_t init_time=clock();
        cout << endl << "Processing ................................" << frame_idx << "." << endl;
        // generate detections
        detections = create_detection(seq_info.detections, frame_idx, args.min_detection_height, args.min_confidence, seq_info.feature_dim);

        // run non-maxima suppression
        vector<vector<float> > boxes;
        vector<float> scores;
        for(vector<Detection>::iterator it = detections.begin(); it!=detections.end(); ++it)
        {
            boxes.push_back(it->to_tlbr());
            scores.push_back(it->confidence_);
        }
        vector<int> pick = nms(boxes, args.nms_max_overlap, scores);
        vector<Detection> nms_detections;
        for(vector<int>::iterator it = pick.begin(); it!=pick.end();++it)
        {
            nms_detections.push_back(detections[*it]);
        }
        detections.clear();
        detections = nms_detections;
        cout << "detections: "<< detections.size() <<endl;

        // update tracker
        clock_t startTime = clock();
        tracker.predict();
        float predictTime = (float)(clock()-startTime)/CLOCKS_PER_SEC;
        tracker.update(detections);
        float totalTime = (float)(clock()-startTime)/CLOCKS_PER_SEC;
        startTime = clock();
        cout << "track_use_time: " <<  totalTime << "; predict/total: "<< predictTime/totalTime << endl;

        if(args.display)
        {
            // update visualization
            cv::Mat image = cv::imread(seq_info.image_file, CV_LOAD_IMAGE_COLOR);

            //draw detections
            // float ratio = seq_info.image_size[0] / detection_area[0];
            float ratio = seq_info.image_size[0] / 4.0;
            draw_detections(&image, detections, ratio);
            cout << "draw detection time: " <<  (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;
            startTime = clock();

            //draw tracks
            draw_tracks(&image, tracker.tracks_, ratio);
            cout << "draw track time: " <<  (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;
            startTime = clock();

            // resize
            cv::imshow("image", image);
            cout << "draw image time: " <<  (float)(clock()-startTime)/CLOCKS_PER_SEC << endl;

            float wait_time = seq_info.update_ms - (float)(clock()-init_time)/CLOCKS_PER_SEC*1000 ;
            wait_time = wait_time>1?(int)(wait_time):1;
            cout << "waite another " << wait_time << " ms" << endl;
            cv::waitKey(wait_time);

        }
        frame_idx ++;
        detections.clear();
    }

}



int main(int argc, char** argv)
{
    Args args;
    vector<string> arg_vec(argv+1, argv+argc);


    for(vector<string>::iterator it = arg_vec.begin(); it!= arg_vec.end(); ++it)
    {
        int pos = it->find("=");
        string arg_string = it->substr(2, pos-2);
        if(arg_string.compare("help") == 0){
            cout << "there is no help, lueluelue" << endl;
            return 0;
        }
        else if(arg_string.compare("sequence_dir") == 0){
            args.sequence_dir = it->substr(pos+1);
        }
        else if(arg_string.compare("detection_file") == 0){
            args.detection_file = it->substr(pos+1);
        }
        else if(arg_string.compare("output_file") == 0){
            args.output_file = it->substr(pos+1);
        }
        else if(arg_string.compare("min_confidence") == 0){
            args.min_confidence = stof(it->substr(pos+1));
        }
        else if(arg_string.compare("min_detection_height") == 0){
            args.min_detection_height = stoi(it->substr(pos+1));
        }
        else if(arg_string.compare("nms_max_overlap") == 0){
            args.nms_max_overlap = stof(it->substr(pos+1));
        }
        else if(arg_string.compare("max_cosine_distance") == 0){
            args.max_cosine_distance = stof(it->substr(pos+1));
        }
        else if(arg_string.compare("nn_budget") == 0){
            args.nn_budget = stoi(it->substr(pos+1));
        }
        else if(arg_string.compare("display") == 0){
            args.display = it->substr(pos+1)=="true";
        }
        else{
            cout << "error: input the right arguments: " << arg_string<< endl;
            return 0;
        }
    }

    run(args);


}
