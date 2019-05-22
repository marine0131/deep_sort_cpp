This is a c++ version of deep_sort(https://github.com/nwojke/deep_sort.git)

#Running the tracker
use the same command as deep_sort python
    ./deep_sort_app \
        --sequence_dir=../MOT16-12 \
        --detection_file=../MOT16-12/MOT16-12.npy \
        --min_confidence=0.6 \
        --min_detection_height=200 \
        --nn_budget=100 \
        --display=true
