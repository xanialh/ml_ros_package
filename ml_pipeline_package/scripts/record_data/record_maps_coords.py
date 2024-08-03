import rospy
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid
import datetime
import numpy as np
import os
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
import yaml

class Ros_recorder:
    def __init__(self,i,date_time_recording,folder_path,ogm_topic,sgm_topic,odom_topic):
        self.i = i
        self.ogm_topic = ogm_topic
        self.sgm_topic = sgm_topic
        self.odom_topic = odom_topic
        self.date_time_recording = date_time_recording
        self.filename = os.path.join(folder_path, str(self.i) + "_recording_" + date_time_recording + ".txt")
        self.file = open(self.filename,"w")
        print("New file made")
        self.recording_flag = True
        self.ogm_sub = None
        self.sgm_sub = None
        self.odom_x = None
        self.odom_y = None

    def set_recording_flag(self,value):
        self.recording_flag = False
        if value == False:
            self.file.close()
            print("file closed")
            if self.ogm_sub is not None:
                self.ogm_sub.unregister()
            if self.sgm_sub is not None:
                self.sgm_sub.unregister()
    
    def s_gridmap_callback(self,msg,appendFile):
        layers_copy = msg.layers.copy()
        # Extract data from the 'social' layer
        social_gridmap_data = None
        for layer, data in zip(layers_copy , msg.data):
            if layer == 'social_heatmap':
                social_gridmap_data = data
                break

        if social_gridmap_data is None:
            rospy.logwarn("No social grid map data found")
            return

        data = social_gridmap_data
        sequence = float(msg.info.header.seq - 1)
        column_index = None
        row_index = None

        dimensions = msg.data[0].layout.dim

        for d in dimensions:
            if d.label == "column_index":
                column_index = d.size
            elif d.label == "row_index":
                row_index = d.size
        data_array = np.array(data.data)
        positions = [float("nan"),float("nan")]

        if self.odom_x is not None and self.odom_y is not None:
            positions[0] = self.odom_x
            positions[1] = self.odom_y

        positions_array = np.array(positions)

        numpy_array = np.insert(data_array,0,positions_array)

        array_with_width = np.insert(numpy_array,0,float(row_index))
        array_with_height = np.insert(array_with_width ,0,float(column_index))

        seq_array = np.insert(array_with_height,0,sequence)

        final_array = np.insert(seq_array,0,float(1))

        np.savetxt(appendFile, [final_array],fmt='%f',delimiter=",",newline='\n')
        print("S WROTE: "+ str(sequence))

    def o_gridmap_callback(self,msg,appendFile):
        data = msg.data
        column_index = msg.info.width
        row_index = msg.info.height

        data_array = np.array(data)
        
        positions = [float("nan"),float("nan")]

        if self.odom_x is not None and self.odom_y is not None:
            positions[0] = self.odom_x
            positions[1] = self.odom_y

        positions_array = np.array(positions)

        numpy_array = np.insert(data_array,0,positions_array)

        array_with_width = np.insert(numpy_array,0,float(column_index))
        array_with_height = np.insert(array_with_width ,0,float(row_index))

        sequence = float(msg.header.seq)

        seq_array = np.insert(array_with_height,0,sequence)

        final_array = np.insert(seq_array,0,float(0))

        np.savetxt(appendFile, [final_array],fmt='%f',delimiter=",",newline='\n')
        print("O WROTE: " + str(sequence))

    def coords_callback(self,msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y

    def record(self):
            self.file.close()
            topics = [self.ogm_topic, self.sgm_topic,self.odom_topic]
 
            self.file = open(self.filename,'a')

            self.ogm_sub = rospy.Subscriber(topics[0], OccupancyGrid, self.o_gridmap_callback,callback_args=self.file)
            self.sgm_sub = rospy.Subscriber(topics[1], GridMap, self.s_gridmap_callback,callback_args=self.file)
            self.coords = rospy.Subscriber(topics[2], Odometry, self.coords_callback)

class dataCollector:
    def __init__(self,folder_path,ogm_topic,sgm_topic,odom_topic,max_files) -> None:
        self.folder_path = folder_path
        self.ogm_topic = ogm_topic
        self.sgm_topic = sgm_topic
        self.odom_topic = odom_topic
        self.recording_flag = True
        self.current_recorder = None
        self.i = 0
        self.max_files = max_files
        rospy.Subscriber("/route_end",Bool,callback=self.endRecorderCallback,queue_size=10)

    def endRecorderCallback(self,msg):
        booleanValue = bool(msg.data)
        self.recording_flag = booleanValue

    def loop(self):
        timeLog = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        while not rospy.is_shutdown():    
            self.i = self.i + 1
            print("ith: " + str(self.i))
            try: 
                if self.i > self.max_files:
                    print("Maximum number of files reached")
                    break
                print("New recorder made")
                self.current_recorder = Ros_recorder(self.i,timeLog,self.folder_path,self.ogm_topic,self.sgm_topic,self.odom_topic)
                self.current_recorder.record()

                while self.recording_flag:
                    rospy.sleep(1)

                print("route end flag flipped")    

                self.current_recorder.set_recording_flag(False)
                print("Recorder flag flipped")
                self.current_recorder = None
                print("Recorder successfully ended")
            except Exception as e:
                print(e)
                if self.current_recorder:
                    self.current_recorder.set_recording_flag(False)
                    print("current recorder flag turned false")
                    self.current_recorder = None  # Clear reference to stopped

def load_config():
    # Load configuration
    try:
        with open("ml_pipeline_package/config/pipelineConfig.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print("Error: Configuration file 'config.yaml' not found!")
    # Handle the error or use default values

def main():
    config_full = load_config()
    config = config_full["record_maps_coords"]
    
# Load configuration

    folder_path = config["folder_path"]
    ogm_topic = "/grid_map_visualization/obstacles"
    sgm_topic = "/smf_move_base_mapper/social_grid_map"
    odom_topic = "/pepper/odom_groundtruth"
    max_files = config["max_files"]

#social grid map has 1 as first element
#obstacle grid map has 0 as first element

    rospy.init_node("gridMapCoordCollector")
    file_maker = dataCollector(folder_path,ogm_topic,sgm_topic,odom_topic,max_files)
    file_maker.loop()


if __name__ == "__main__":
    main()