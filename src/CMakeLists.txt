# eigen3
FIND_PACKAGE(Eigen3)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})


ADD_LIBRARY(kalman_filter kalman_filter.cpp)                               
TARGET_LINK_LIBRARIES(kalman_filter
    )

ADD_LIBRARY(detection detection.cpp)                               
TARGET_LINK_LIBRARIES(detection
    )

ADD_LIBRARY(track track.cpp)                               
TARGET_LINK_LIBRARIES(track
    kalman_filter
    detection
    )

ADD_LIBRARY(hungarian_alg hungarian_alg.cpp)                               
TARGET_LINK_LIBRARIES(hungarian_alg
    )

ADD_LIBRARY(linear_assignment linear_assignment.cpp)                               
TARGET_LINK_LIBRARIES(linear_assignment
    track
    hungarian_alg
    )

ADD_LIBRARY(nn_matching nn_matching.cpp)                               
TARGET_LINK_LIBRARIES(nn_matching
    )

ADD_LIBRARY(iou_matching iou_matching.cpp)
TARGET_LINK_LIBRARIES(iou_matching
    track)

ADD_LIBRARY(tracker tracker.cpp)
TARGET_LINK_LIBRARIES(tracker
    linear_assignment
    nn_matching
    iou_matching)

ADD_LIBRARY(nms nms.cpp)
TARGET_LINK_LIBRARIES(nms
    )

ADD_EXECUTABLE(deep_sort_app app_util/deep_sort_app.cpp)
TARGET_LINK_LIBRARIES(deep_sort_app
    tracker
    detection
    cnpy
    nms
    ${OpenCV_LIBS}
    )

ADD_EXECUTABLE(deep_sort_app_pcl app_util/deep_sort_app_pcl.cpp)
TARGET_LINK_LIBRARIES(deep_sort_app_pcl
    tracker
    detection
    cnpy
    nms
    ${OpenCV_LIBS}
    )

ADD_EXECUTABLE(deep_sort_test app_util/deep_sort_test.cpp)
TARGET_LINK_LIBRARIES(deep_sort_test
    tracker
    detection
    nms
    ${OpenCV_LIBS}
    )
