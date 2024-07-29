import rospy
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid
import datetime
import numpy as np
import os
from std_msgs.msg import Bool
import yaml

class rosRecorder:
    def __init__(self,i,datetimeRecording,folder_path,ogm_topic,sgm_topic):
        self.i = i
        self.ogm_topic = ogm_topic
        self.sgm_topic = sgm_topic
        self.datetimeRecording = datetimeRecording
        self.fileName = os.path.join(folder_path, str(self.i) + "_recording" + datetimeRecording + ".txt")
        self.file = open(self.fileName,"w")
        print("New file made")
        self.recordingFlag = True
        self.ogmSub = None
        self.sgmSub = None

    def setRecordingFlag(self,value):
        self.recordingFlag = False
        if value == False:
            self.file.close()
            print("file closed")
            if self.ogmSub is not None:
                self.ogmSub.unregister()
            if self.sgmSub is not None:
                self.sgmSub.unregister()
    
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
        social_heatmap_data = None
        for layer, data in zip(layers_copy , msg.data):
            if layer == 'social_heatmap':
                social_heatmap_data = data
                break

        if social_heatmap_data is None:
            rospy.logwarn("No social heat map data found")
            return

        data = social_heatmap_data
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

        arrayWithWidth = np.insert(numpy_array,0,float(row_index))
        arrayWithHeight = np.insert(arrayWithWidth ,0,float(column_index))

        seq_array = np.insert(arrayWithHeight,0,sequence)

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

        arrayWithWidth = np.insert(numpy_array,0,float(column_index))
        arrayWithHeight = np.insert(arrayWithWidth ,0,float(row_index))

        sequence = float(msg.header.seq)

        seq_array = np.insert(arrayWithHeight,0,sequence)

        final_array = np.insert(seq_array,0,float(0))

        np.savetxt(appendFile, [final_array],fmt='%f',delimiter=",",newline='\n')
        print("O WROTE: " + str(sequence))
    
    def record(self):
            self.file.close()
            topics = [self.ogm_topic, self.sgm_topic]

            self.file = open(self.fileName,'a')

            self.ogmSub = rospy.Subscriber(topics[0], OccupancyGrid, self.o_gridmap_callback,callback_args=self.file)
            self.sgmSub = rospy.Subscriber(topics[1], GridMap, self.s_gridmap_callback,callback_args=self.file)



class dataCollector:
    def __init__(self,folder_path,ogm_topic,sgm_topic) -> None:
        self.folder_path = folder_path
        self.ogm_topic = ogm_topic
        self.sgm_topic = sgm_topic
        self.recordingFlag = True
        self.currentRecorder = None
        self.i = 0
        rospy.Subscriber("/routeEnd",Bool,callback=self.endRecorderCallback,queue_size=10)

    def endRecorderCallback(self,msg):
        booleanValue = bool(msg.data)
        self.recordingFlag = booleanValue

    def loop(self):
        timeLog = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        while not rospy.is_shutdown():    
            self.i = self.i + 1
            print("ith: " + str(self.i))
            try: 
                print("New recorder made")
                self.currentRecorder = rosRecorder(self.i,timeLog,self.folder_path,self.ogm_topic,self.sgm_topic)
                self.currentRecorder.record()

                while self.recordingFlag:
                    rospy.sleep(1)

                print("route end flag flipped")    

                self.currentRecorder.setRecordingFlag(False)
                print("Recorder flag flipped")
                self.currentRecorder = None
                print("Recorder successfully ended")
            except Exception as e:
                print(e)
                if self.currentRecorder:
                    self.currentRecorder.setRecordingFlag(False)
                    print("current recorder flag turned false")
                    self.currentRecorder = None  # Clear reference to stopped


def loadConfig():
    # Load configuration
    try:
        with open("ml_pipeline_package/config/pipelineConfig.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print("Error: Configuration file 'config.yaml' not found!")
    # Handle the error or use default values

def main():
    configFull = loadConfig()
    config = configFull["record_maps"]

    folder_path = config["folder_path"]
    ogm_topic = config["ogm_topic"]
    sgm_topic = config["sgm_topic"]

    #social grid map has 1 as first element
    #obstacle grid map has 0 as first element

    rospy.init_node("gridMapCollector")
    fileMaker = dataCollector(folder_path,ogm_topic,sgm_topic)
    fileMaker.loop()

if __name__ == "__main__":
    main()