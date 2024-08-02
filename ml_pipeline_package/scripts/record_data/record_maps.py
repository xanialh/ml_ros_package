import rospy
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid
import datetime
import numpy as np
import os
from std_msgs.msg import Bool
import yaml

class Ros_recorder:
    def __init__(self,i,datetime_recording,folder_path,ogm_topic,sgm_topic):
        self.i = i
        self.ogm_topic = ogm_topic
        self.sgm_topic = sgm_topic
        self.datetime_recording = datetime_recording
        self.filename = os.path.join(folder_path, str(self.i) + "_recording_" + datetime_recording + ".txt")
        self.file = open(self.filename,"w")
        print("New file made")
        self.recording_flag = True
        self.ogm_sub = None
        self.sgm_sub = None

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
        '''
    info.header.seq
    info.header.stamp.secs
    info.header.stamp.nsecs
    info.header.frame_id
    info.resolution
    info.length_x
    info.length_y
    info.pose.position.x
    info.pose.position.y
    info.pose.position.z
    info.pose.orientation.x
    info.pose.orientation.y
    info.pose.orientation.z
    info.pose.orientation.w
    layers
    basic_layers
    data
    outer_start_index
    inner_start_index
    '''
        
        layers_copy = msg.layers.copy()
        # Extract data from the 'social' layer
        social_gridmap_data = None
        for layer, data in zip(layers_copy , msg.data):
            if layer == 'social_gridmap':
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
        numpy_array = np.array(data.data)

        array_with_width = np.insert(numpy_array,0,float(row_index))
        array_with_height = np.insert(array_with_width ,0,float(column_index))

        seq_array = np.insert(array_with_height,0,sequence)

        final_array = np.insert(seq_array,0,float(1))

        np.savetxt(appendFile, [final_array],fmt='%f',delimiter=",",newline='\n')
        print("S WROTE: "+ str(sequence))

    def o_gridmap_callback(self,msg,appendFile):
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

        numpy_array = np.array(data)

        array_with_width = np.insert(numpy_array,0,float(column_index))
        array_with_height = np.insert(array_with_width ,0,float(row_index))

        sequence = float(msg.header.seq)

        seq_array = np.insert(array_with_height,0,sequence)

        final_array = np.insert(seq_array,0,float(0))

        np.savetxt(appendFile, [final_array],fmt='%f',delimiter=",",newline='\n')
        print("O WROTE: " + str(sequence))
    
    def record(self):
            self.file.close()
            topics = [self.ogm_topic, self.sgm_topic]

            self.file = open(self.filename,'a')

            self.ogm_sub = rospy.Subscriber(topics[0], OccupancyGrid, self.o_gridmap_callback,callback_args=self.file)
            self.sgm_sub = rospy.Subscriber(topics[1], GridMap, self.s_gridmap_callback,callback_args=self.file)



class Data_collector:
    def __init__(self,folder_path,ogm_topic,sgm_topic,max_files) -> None:
        self.folder_path = folder_path
        self.ogm_topic = ogm_topic
        self.sgm_topic = sgm_topic
        self.recording_flag = True
        self.current_recorder = None
        self.i = 0
        self.max_files = max_files
        rospy.Subscriber("/route_end",Bool,callback=self.end_recorder_callback,queue_size=10)

    def end_recorder_callback(self,msg):
        boolean_value = bool(msg.data)
        self.recording_flag = boolean_value

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
                self.current_recorder = Ros_recorder(self.i,timeLog,self.folder_path,self.ogm_topic,self.sgm_topic)
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
    config = config_full["record_maps"]

    folder_path = config["folder_path"]
    ogm_topic = config["ogm_topic"]
    sgm_topic = config["sgm_topic"]
    max_files = config["max_files"]

    #social grid map has 1 as first element
    #obstacle grid map has 0 as first element

    rospy.init_node("gridMapCollector")
    file_maker = Data_collector(folder_path,ogm_topic,sgm_topic,max_files)
    file_maker.loop()

if __name__ == "__main__":
    main()