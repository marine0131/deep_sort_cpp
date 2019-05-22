#include <deep_sort_app.h>
#include <cnpy.h>

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
void draw_tracks(cv::Mat* im, vector<Track> tracks)
{
    for(vector<Track>::iterator it=tracks.begin(); it!= tracks.end(); ++it)
    {
        if(!it->is_confirmed()||it->time_since_update_>0)
            continue;
        cv::Scalar color = create_unique_color(it->track_id_);
        
        //self.viewer.rectangle(*track.to_tlwh().astype(np.int), label=str(track.track_id))
        cv::Point pt1, pt2;
        vector<float> tlwh = it->to_tlwh();
        pt1.x = (int)tlwh[0];
        pt1.y = (int)tlwh[1];
        pt2.x = (int)(tlwh[0] + tlwh[2]);
        pt2.y = (int)(tlwh[1] + tlwh[3]);
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
        cv::putText(*im, label, center, cv::FONT_HERSHEY_PLAIN,1, (255, 255, 255), 2);
    }
}

void draw_detections(cv::Mat* im, vector<Detection> detections)
{
    for(vector<Detection>::iterator it = detections.begin(); it != detections.end(); ++it)
    {
        cv::Point pt1, pt2;
        pt1.x = (int)it->tlwh_[0];
        pt1.y = (int)it->tlwh_[1];
        pt2.x = (int)(it->tlwh_[0]+it->tlwh_[2]);
        pt2.y = (int)(it->tlwh_[1]+it->tlwh_[3]);
        // cout << "(" << pt1.x<<","<<pt1.y<<") ("<< pt2.x << ", " << pt2.y<< ")" << endl;

        cv::rectangle(*im, pt1, pt2, cv::Scalar(255, 255, 255), 2);
    }
}

vector<Detection> create_detection(vector<vector<float> > detection_mat, int frame_idx, int min_height, float min_confidence, int feature_dim)
{
    vector<Detection> detections;

    // cout << "frame_id: "<< frame_idx << " DetectionIdx: " << DetectionIdx<< endl;
    size_t i = 0;
    for(i = DetectionIdx; i < detection_mat.size(); ++i)
    {
        if((int)detection_mat[i][0] > frame_idx)
            break;
        if((int)detection_mat[i][0] == frame_idx)
        {
            vector<float> bbox(4);
            copy(detection_mat[i].begin()+2, detection_mat[i].begin()+6, bbox.begin());
            float confidence = detection_mat[i][6];
            vector<float> feature(feature_dim);
            copy(detection_mat[i].begin()+10, detection_mat[i].end(), feature.begin());
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

vector<string> get_line(string filename)
{
    ifstream in(filename);
    vector<string> lines;

    if(!in.rdbuf()->is_open())
    {
        cout << "file " << filename << " not  exits"<<endl;
        exit(1);
    }

    string tmp;
    while(getline(in, tmp))
    {
        lines.push_back(tmp);
    }

    sort(lines.begin(), lines.end());
    return lines;
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
     *
     *
     *
     */

    seq_info->sequence_name = sequence_dir;

    string image_dir = sequence_dir+"/img1";
    vector<string> image_file_list = get_line(sequence_dir+"/filenames.txt");
    for(vector<string>::iterator it = image_file_list.begin(); it != image_file_list.end(); ++it)
    {
        int num = stoi(it->substr(0, it->find(".")));
        seq_info->image_filenames.insert(pair<int, string>(num, image_dir+"/"+*it));
    }
    // for(map<int, string>::iterator it = image_filenames.begin(); it != image_filenames.end(); ++it)
    //     cout << it->first << ": " << it->second <<endl;
    cout << "total image: " << seq_info->image_filenames.size() << endl;
    if(detection_file == "")
    {
        cout << "error: need detection file, exiting..." << endl;
        exit(1);
    }
    cnpy::NpyArray arr = cnpy::npy_load(detection_file);
    cout << "detection shape: (" << arr.shape[0]  << ", "<< arr.shape[1] << ")"<< endl;
    cout << "detection word size: " << arr.word_size << endl;
    double* loaded_data = arr.data<double>();
    vector<vector<float> > detections(arr.shape[0], vector<float>(arr.shape[1], 0));
    for(size_t i = 0; i< arr.shape[0]; ++i)
        for(size_t j = 0; j < arr.shape[1]; ++j)
            detections[i][j] = (float)(loaded_data[i*arr.shape[1]+j]);
    seq_info->detections = detections;
    //        cout << loaded_data[i*arr.shape[1]+j] << ", ";
    //     cout << endl;
    // }
    seq_info->groundtruth = "";
    
    const string im_file = seq_info->image_filenames.begin()->second;
    cv::Mat im = cv::imread(im_file, CV_LOAD_IMAGE_COLOR);
    seq_info->image_size.push_back(im.cols);
    seq_info->image_size.push_back(im.rows);
    // cv::imshow("window", im);
    // cv::waitKey(0);
    //
    seq_info->min_frame_idx = seq_info->image_filenames.begin()->first;
    seq_info->max_frame_idx = seq_info->image_filenames.end()->first;
     cout << "frame_idx: " << seq_info->min_frame_idx << "~" << seq_info->max_frame_idx << endl;
    
    //string info_filename = sequence_dir + "/seqInfo.ini";
    seq_info->update_ms = 1000/30; //hz

    seq_info->feature_dim = arr.shape[1] -10;
    cout << "feature dim: " << seq_info->feature_dim << endl;

}



void run(Args args)
{
    SeqInfo seq_info;
    gather_sequence_info(&seq_info, args.sequence_dir, args.detection_file);
    cout <<  "image_size: " << seq_info.image_size[0] << ", " << seq_info.image_size[1] << endl;

    // NearestNeighborDistanceMetric metric("cosine", args.max_cosine_distance, args.nn_budget);
    string metric = "cosine";
    Tracker tracker(metric, args.max_cosine_distance);
    int frame_idx = seq_info.min_frame_idx;
    int last_idx = seq_info.max_frame_idx;
    vector<Detection> detections;
    while(frame_idx < last_idx)
    {
        cout << "Processing " << seq_info.image_filenames.find(frame_idx)->second << "." << endl;
        // generate detections
        detections = create_detection(seq_info.detections, frame_idx, args.min_detection_height, args.min_confidence, seq_info.feature_dim);

        // run non-maxima suppression

        // update tracker
        clock_t startTime = clock();
        tracker.predict();
        float predictTime = (float)(clock()-startTime)/CLOCKS_PER_SEC;
        tracker.update(detections);
        float totalTime = (float)(clock()-startTime)/CLOCKS_PER_SEC;
        cout << "track_use_time: " <<  totalTime << "; predict/total: "<< predictTime/totalTime << endl;

        if(args.display)
        {
            // update visualization
            cv::Mat image = cv::imread(seq_info.image_filenames.find(frame_idx)->second, CV_LOAD_IMAGE_COLOR);

            //draw detections
            draw_detections(&image, detections);

            //draw tracks
            draw_tracks(&image, tracker.tracks_);

            // resize
            float aspect_ratio = (float)(seq_info.image_size[1]) / (float)(seq_info.image_size[0]);
            cv::Size dsize = cv::Size(1024, (int)(aspect_ratio*1024));
            cv::resize(image, image, dsize, 0, 0, cv::INTER_AREA);
            cv::imshow("image", image);
            cv::waitKey(1);

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
