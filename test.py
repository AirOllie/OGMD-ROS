#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
import numpy as np


image_dir = '/home/nanostring/catkin_ws/src/ogmd/src/AMG.pgm'

rospy.init_node('image_publisher', anonymous=True)
image_pub = rospy.Publisher("/occupancy_map", Image, queue_size=10)


def publish_image():
    # Load an image using OpenCV
    cv_image = cv2.imread(image_dir)
    if cv_image is not None:
        try:
            # Create a new Image message
            ros_image = Image()
            ros_image.header.stamp = rospy.Time.now()
            ros_image.header.frame_id = "frame"
            ros_image.height = cv_image.shape[0]
            ros_image.width = cv_image.shape[1]
            ros_image.encoding = 'bgr8'
            ros_image.is_bigendian = False
            ros_image.step = cv_image.shape[1] * cv_image.shape[2]
            ros_image.data = cv_image.tobytes()

            # Publish the image
            image_pub.publish(ros_image)
            rospy.loginfo("Image has been published.")
        except Exception as e:
            rospy.logerr("Failed to publish image: %s", e)
    else:
        rospy.logerr("Failed to load image.")


if __name__ == '__main__':
    try:
        rate = rospy.Rate(1)  # 1 Hz
        while not rospy.is_shutdown():
            publish_image()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
