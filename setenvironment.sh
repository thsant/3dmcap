#!/bin/bash
# Inform Mayavi to use WX
export ETS_TOOLKIT=wx
# Location of our Python wrapper to ORB-SLAM
export PYTHONPATH=$PYTHONPATH:./3rdparty/ORB_SLAM2/python
# ORB-SLAM lib location
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./3rdparty/ORB_SLAM2/lib:./3rdparty/DBoW2/lib:./3rdparty/g2o/lib
# Set fixed focus
v4l2-ctl -d /dev/video1 -c focus_auto=0
v4l2-ctl -d /dev/video1 -c focus_absolute=0
