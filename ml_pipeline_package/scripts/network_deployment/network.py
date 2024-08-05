import rospy
from grid_map_msgs.msg import GridMap, GridMapInfo
from nav_msgs.msg import OccupancyGrid
import datetime
import numpy as np
import os
from std_msgs.msg import Bool
import yaml
from ml_networks import FCN
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# Load configuration
try:
  with open("/ml_pipeline_package/config/pipelineConfig.yaml", "r") as f:
    config = yaml.safe_load(f)
except FileNotFoundError:
  print("Error: Configuration file 'config.yaml' not found!")
  # Handle the error or use default values

folder_path = config["folder_path"]
ogm_topic = config["ogm_topic"]
sgm_topic = config["sgm_topic"]
model_path = config["model_path"]
training_img_size = config["training_img_size"]

model = FCN.FCN()

model.load_state_dict(torch.load(model_path))

def o_gridmap_callback(msg,model):
    '''
header.seq
header.stamp.secs
header.stamp.nsecs
header.frame_id
info.map_load_time.secs
info.map_load_time.nsecs
info.resolution
info.width
info.height
info.origin.position.x
info.origin.position.y
info.origin.position.z
info.origin.orientation.x
info.origin.orientation.y
info.origin.orientation.z
info.origin.orientation.w
data
'''
    data = msg.data
    column_index = msg.info.width
    row_index = msg.info.height
    seq = msg.header.seq

    numpy_array = np.array(data,dtype=np.float32)
    numpy_array = numpy_array.reshape(row_index,column_index)
    # Resize to 128x128 using OpenCV (adjust interpolation method if needed)
    resized_array = cv2.resize(numpy_array, tuple(training_img_size), interpolation=cv2.INTER_AREA)
    image_tensor = torch.from_numpy(resized_array)

    output = model(image_tensor.unsqueeze(0))

    output = torch.argmax(output, dim=1)

    np_low = np.array(output)

    low_msg = OccupancyGrid()

    low_msg.header.seq = seq
    low_msg.info.height = 128
    low_msg.info.width = 128
    low_msg.data = np_low.tolist()

    sgmPub.publish(low_msg)

rospy.init_node("mlNetwork")
ogmSub = rospy.Subscriber(ogm_topic, OccupancyGrid, o_gridmap_callback,callback_args=model)
sgmPub = rospy.Publisher("/ml_sgm",OccupancyGrid,queue_size=10)

rospy.spin()