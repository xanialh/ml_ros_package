import time
import roslaunch
import rospy
import random
import time
import roslaunch
import rospy
from smf_move_base_msgs.msg import Goto2DActionResult
from rosgraph_msgs.msg import Clock
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Bool
import yaml

# Load configuration
try:
  with open("/home/xanial/FINAL_YEAR_PROJECT/ml_ros_package/ml_pipeline_package/config/pipelineConfig.yaml", "r") as f:
    configFull = yaml.safe_load(f)
    # load entire config
except FileNotFoundError:
  print("Error: Configuration file 'config.yaml' not found!")
  # Handle the error or use default values

config = configFull["training_route"]
# load just training route config

# set up config
map = config["map"]
num_waypoints = config["num_waypoints"]
waypoint_topic = config["waypoint_topic"]
goal_result_topic = config["goal_result_topic"]
starting_position = config["starting_position"]
max_route_time = config["max_route_time"]
start_position = random.choice(starting_position)

PROCESS_GENERATE_RUNNING = True

# used to place robot
version_choice = map
start_position = start_position
goal_position = start_position
waypoints = []

class ProcessListener(roslaunch.pmon.ProcessListener):
    """keeps track of the process from launch file"""

    global PROCESS_GENERATE_RUNNING

    def process_died(self, name, exit_code):
        global PROCESS_GENERATE_RUNNING
        PROCESS_GENERATE_RUNNING = False
        rospy.logwarn("%s died with code %s", name, exit_code)


def init_launch(launchfile, process_listener, version, start_position, goal_position):
    """initiates launch file and runs it"""
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    launch = roslaunch.parent.ROSLaunchParent(
        uuid,
        [
            (
                launchfile,
                [
                    "version:=" + version,
                    "x_pos:=" + str(start_position[0]),
                    "y_pos:=" + str(start_position[1]),
                    "x_goal:=" + str(goal_position[0]),
                    "y_goal:=" + str(goal_position[1]),
                ],
            )
        ],
        process_listeners=[process_listener],
    )
    return launch

rospy.init_node("smf_nav_stack_tests_launcher")

LAUNCH_FILE = "/home/xanial/ros/noetic/system/src/pepper_social_nav_tests/launch/smf_nav_stack_test.launch"
launch = init_launch(
    LAUNCH_FILE, ProcessListener(), version_choice, start_position, goal_position)
launch.start()

goal_reached = False

# takes a random sample from entire list of waypoints
def subListWaypoints(aList,num):
    return random.sample(aList,num)

# goal reached function
def goal_reached_callback(msg):
    global goal_reached
    if msg.result.success:
        goal_reached = True

# gets waypoints and sends them to rosparam
def waypoint_callback(msg):
    counter = 0
    for marker in msg.markers:
        counter = counter + 1
        if counter % 3 == 0:
            wp = [marker.pose.position.x,marker.pose.position.y]
            waypoints.append(wp)

    rospy.set_param("waypointList",waypoints)
    waypoint_listener.unregister()

# waypoint listener node
waypoint_listener = rospy.Subscriber(
    waypoint_topic,
    MarkerArray,
    waypoint_callback
)

# goal listener node
goal_listener = rospy.Subscriber(
    goal_result_topic,
    Goto2DActionResult,
    goal_reached_callback,
    queue_size=1,
)

# max time ros param
rospy.set_param("/metrics_recorder_node/max_test_time",max_route_time)
current_time = 0
flag = True

# publiser for recording data
recordPub = rospy.Publisher("/recordFlag", Bool,queue_size=10)

msgEnd = Bool()
msgEnd.data = False
recordPub.publish(msgEnd)

# main loop
while (flag):

    # start recording maps
    msgStart = Bool()
    msgStart.data = True
    recordPub.publish(msgStart)

    # timing
    current_time_msg = rospy.wait_for_message("clock", Clock)
    current_time = current_time_msg.clock.secs
    max_test_time = rospy.get_param("/metrics_recorder_node/max_test_time")

    # goal reached branch
    if goal_reached or (current_time>max_test_time):
        

        if len(waypoints) == 0:
            flag = False
            break
            # no more waypoints
        elif len(waypoints) > num_waypoints:
            waypoints = subListWaypoints(waypoints,num_waypoints)
            # set waypoints for robot

        # raise time to allow robot to reach next waypoint
        rospy.set_param("/metrics_recorder_node/max_test_time",max_route_time+current_time)
        
        # set new waypoint
        new_goal = waypoints.pop(0)

        # set waypoints ros param
        rospy.set_param("waypointList",waypoints)

        # set new end position
        new_x = new_goal[0]
        new_y = new_goal[1]

        # set new end position (rosparam)
        rospy.set_param("/smf_nav_stack_requester/x_goal",new_x)
        rospy.set_param("/smf_nav_stack_requester/y_goal",new_y)

        # set goal reached flag as false
        goal_reached = False

    rospy.sleep(1)

rospy.sleep(1)

# stop recording 
recordPub.publish(msgEnd)

#end
launch.shutdown()