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
  with open("/home/xanial/FINAL_YEAR_PROJECT/ml_ros_package/ml_pipeline_package/config/config_training_route.yaml", "r") as f:
    config = yaml.safe_load(f)
except FileNotFoundError:
  print("Error: Configuration file 'config.yaml' not found!")
  # Handle the error or use default values

map = config["map"]
num_waypoints = config["num_waypoints"]
waypoint_topic = config["waypoint_topic"]
goal_result_topic = config["goal_result_topic"]
starting_position = config["starting_position"]
start_position = random.choice(starting_position)

PROCESS_GENERATE_RUNNING = True

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
    """initiates launch file runs it"""
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

def subListWaypoints(aList,num):
    return random.sample(aList,num)

def goal_reached_callback(msg):
    global goal_reached
    if msg.result.success:
        goal_reached = True

def waypoint_callback(msg):
    counter = 0
    for marker in msg.markers:
        counter = counter + 1
        if counter % 3 == 0:
            wp = [marker.pose.position.x,marker.pose.position.y]
            waypoints.append(wp)

    rospy.set_param("waypointList",waypoints)
    waypoint_listener.unregister()

waypoint_listener = rospy.Subscriber(
    waypoint_topic,
    MarkerArray,
    waypoint_callback
)

goal_listener = rospy.Subscriber(
    goal_result_topic,
    Goto2DActionResult,
    goal_reached_callback,
    queue_size=1,
)

time.sleep(2)

rospy.set_param("/metrics_recorder_node/max_test_time",400)
current_time = 0
flag = True

bagRecordPub = rospy.Publisher("/recordFlag", Bool,queue_size=10)

msgEnd = Bool()
msgEnd.data = False
bagRecordPub.publish(msgEnd)

time.sleep(2)

while (flag):
    msgStart = Bool()
    msgStart.data = True
    bagRecordPub.publish(msgStart)
    rospy.loginfo("Recording grid maps")
    current_time_msg = rospy.wait_for_message("clock", Clock)
    current_time = current_time_msg.clock.secs
    max_test_time = rospy.get_param("/metrics_recorder_node/max_test_time")
    
    if goal_reached or (current_time>max_test_time):
        if len(waypoints) == 0:
            flag = False
            rospy.loginfo("STAR STAR STAR STAR STAR STAR STAR STAR STAR STAR STAR STAR STAR STAR STAR STAR STAR STAR ")
            print("**********************************************************************************************************************************")
            break
        elif len(waypoints) > num_waypoints:
            waypoints = subListWaypoints(waypoints,num_waypoints)
        rospy.set_param("/metrics_recorder_node/max_test_time",400+current_time)
        new_goal = waypoints.pop(0)
        rospy.set_param("waypointList",waypoints)
        new_x = new_goal[0]
        new_y = new_goal[1]
        rospy.set_param("/smf_nav_stack_requester/x_goal",new_x)
        rospy.set_param("/smf_nav_stack_requester/y_goal",new_y)
        goal_reached = False

    rospy.sleep(1)

rospy.sleep(1)
bagRecordPub.publish(msgEnd)
launch.shutdown()