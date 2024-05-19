#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
from PIL import Image as PilImage
import torch
import torchvision.transforms as transforms
from models.se_resnet import se_resnet20

device = torch.device('cpu')
model_dir = '/home/nanostring/catkin_ws/src/ogmd/src/best_acc_ckpt.pth'
image_dir = '/home/nanostring/catkin_ws/src/ogmd/src/AMG.pgm'

def load_model():
    model = se_resnet20(num_classes=1).to(device)
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

model = load_model()
output_pub = rospy.Publisher('map_quality', String, queue_size=10)
last_image = None

def image_callback(msg):
    global last_image
    if msg.encoding == 'rgb8':
        dtype = np.uint8
        channels = 3
    elif msg.encoding == 'bgr8':
        dtype = np.uint8
        channels = 3
    elif msg.encoding == 'mono8':
        dtype = np.uint8
        channels = 1
    else:
        rospy.logerr("Unsupported image encoding: %s" % msg.encoding)
        return

    img_array = np.frombuffer(msg.data, dtype=dtype)
    if channels == 3:
        img_array = img_array.reshape((msg.height, msg.width, channels))
        last_image = PilImage.fromarray(img_array, 'RGB')
    elif channels == 1:
        img_array = img_array.reshape((msg.height, msg.width))
        last_image = PilImage.fromarray(img_array, 'L')
    if last_image is not None:
        rospy.loginfo('Using real time occupancy map from ROS topic')

def process_image():
    global last_image
    if last_image is None:
        rospy.logwarn("No image received from 'occupancy_map' topic, using local map {} instead".format(image_dir))
        try:
            last_image = PilImage.open(image_dir)
        except IOError:
            rospy.logerr('No image received and default image not found')
            return

    image = last_image if last_image.mode == 'RGB' else last_image.convert('RGB')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)[0]
        tag = 'Normal' if output[0] > 0.5 else 'Abnormal'
        output_pub.publish(tag)

if __name__ == '__main__':
    rospy.init_node('ogmd')
    rospy.Subscriber('/occupancy_map', Image, image_callback)
    timer = rospy.Timer(rospy.Duration(1), lambda event: process_image())
    rospy.spin()
