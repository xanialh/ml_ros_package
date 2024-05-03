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

PROCESS_GENERATE_RUNNING = True

version_choice = "office"
start_position = [-19.78, 21.57]
goal_position = [-19.78, 21.57]
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

LAUNCH_FILE = "/home/danielhixson/ros/noetic/system/src/pepper_social_nav_tests/launch/smf_nav_stack_test.launch"
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
    "/pedsim_visualizer/waypoints",
    MarkerArray,
    waypoint_callback
)

goal_listener = rospy.Subscriber(
    "/smf_goto_action/result",
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
    rospy.loginfo("Bag is recording")
    current_time_msg = rospy.wait_for_message("clock", Clock)
    current_time = current_time_msg.clock.secs
    max_test_time = rospy.get_param("/metrics_recorder_node/max_test_time")
    
    if goal_reached or (current_time>max_test_time):
        if len(waypoints) == 0:
            flag = False
            break
        elif len(waypoints) > 1:
            waypoints = subListWaypoints(waypoints,1)
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