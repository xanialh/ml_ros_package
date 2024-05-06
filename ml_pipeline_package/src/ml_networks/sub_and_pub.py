import rospy
from grid_map_msgs.msg import GridMap, GridMapInfo
from nav_msgs.msg import OccupancyGrid
import datetime
import numpy as np
import os
from std_msgs.msg import Bool
import yaml
from HM_FCNv2 import SocialHeatMapFCN
import torch
import torch.nn as nn
import torch.optim as optim
import cv2


# Load configuration
try:
  with open("config/config_record_maps.yaml", "r") as f:
    config = yaml.safe_load(f)
except FileNotFoundError:
  print("Error: Configuration file 'config.yaml' not found!")
  # Handle the error or use default values

folder_path = config["folder_path"]
ogm_topic = config["ogm_topic"]
sgm_topic = config["sgm_topic"]

model = SocialHeatMapFCN()
model_path = "data/trained_models/office/FCNv2MODEL.pt"

model.load_state_dict(torch.load(model_path))

model.eval()

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
    resized_array = cv2.resize(numpy_array, (128, 128), interpolation=cv2.INTER_AREA)
    image_tensor = torch.from_numpy(resized_array)

    output = model(image_tensor.unsqueeze(0))

    low_tensor = output[0]
    medium_tensor = output[1]
    high_tensor = output[2]

    flattened_low = low_tensor.reshape(-1)
    flattened_medium = medium_tensor.reshape(-1)
    flattened_high = high_tensor.reshape(-1)

    np_low = flattened_low.detach()
    np_medium = flattened_medium.detach()
    np_high = flattened_high.detach()

    np_low = np.array(np_low)
    np_low = np.floor(np_low * 100)
    np_low = np_low.astype(int)

    np_medium = np.array(np_medium)
    np_medium= np.floor(np_medium * 100)
    np_medium = np_medium.astype(int)


    np_high = np.array(np_high)
    np_high  = np.floor(np_high * 100)
    np_high = np_high.astype(int)

    low_msg = OccupancyGrid()
    medium_msg = OccupancyGrid()
    high_msg = OccupancyGrid()

    low_msg.header.seq = seq
    low_msg.info.height = 128
    low_msg.info.width = 128
    low_msg.data = np_low.tolist()

    medium_msg.header.seq = seq
    medium_msg.info.height = 128
    medium_msg.info.width = 128
    medium_msg.data = np_medium.tolist()

    high_msg.header.seq = seq
    high_msg.info.height = 128
    high_msg.info.width = 128
    high_msg.data = np_high.tolist()

    sgmPubLow.publish(low_msg)
    sgmPubMedium.publish(medium_msg)
    sgmPubHigh.publish(high_msg)

rospy.init_node("mlNetwork")
ogmSub = rospy.Subscriber(ogm_topic, OccupancyGrid, o_gridmap_callback,callback_args=model,)
sgmPubLow = rospy.Publisher("/ml_sgm_low",OccupancyGrid,queue_size=10)
sgmPubMedium = rospy.Publisher("/ml_sgm_medium",OccupancyGrid,queue_size=10)
sgmPubHigh = rospy.Publisher("/ml_sgm_high",OccupancyGrid,queue_size=10)

rospy.spin()