This is a c++ version of deep_sort(https://github.com/nwojke/deep_sort.git)

eigen3 opencv3.3 needed

# Running the tracker
use the same command as deep_sort python
```
    ./deep_sort_app \
        --sequence_dir=../MOT16-12 \
        --detection_file=../MOT16-12/MOT16-12.npy \
        --min_confidence=0.6 \
        --min_detection_height=200 \
        --nn_budget=100 \
        --display=true
```

# Running pcl tracker
use pcl result replace image, track the object in point cloud
```
    
    ./deep_sort_app_pcl \
        --detection_file=../data/detections.txt \
        --min_confidence=0.6 \
        --min_detection_height=0.1 \ 
        --nn_budget=100 \
        --display=true
```

# API
for implementation, I write a interface for pcl use, use command below
    for test.
```

    ./deep_sort_test
```

