## Create ROS package
```
catkin_create_pkg ogmd rospy std_msgs sensor_msgs
```

## Copy the files from the repository to the package
Make sure that **models** and **best_acc_ckpt.pth** are in the same directory as the **test_map_quality.py** file.

## Add execution permission to the script
```
chmod +x test_map_quality.py
```

## Build the package
```
cd ~/catkin_ws
catkin_make
```

## Make sure roscore is running
```
roscore
```

## Run map quality detection node
Before running the code, either publish the occupancy grid map to the topic **/occupancy_map** or save it to **occupancy_map.pgm** in the same directory as the **test_map_quality.py** file, then run
```
rosrun ogmd test_map_quality.py
```
the detected map quality, either **Normal** or **Abnormal**, will be published to **/map_quality** topic
