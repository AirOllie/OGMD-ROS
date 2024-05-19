## Create ROS package
```
catkin_create_pkg ogmd rospy std_msgs sensor_msgs
```

## Copy the files from the repository to the package

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
```
rosrun ogmd test_map_quality.py
```
