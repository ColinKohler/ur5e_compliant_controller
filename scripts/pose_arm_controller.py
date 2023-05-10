#! /usr/bin/env python
import rospy
import rospkg
import numpy as np
from copy import deepcopy
import copy
import time
import Queue
from scipy.interpolate import InterpolatedUnivariateSpline
from ros_numpy import numpify

from std_msgs.msg import Float64MultiArray, Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped
from moveit_msgs.msg import RobotState, Constraints, JointConstraint
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest, GetPositionFKResponse

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

class ur5e_position_controller(object):
    def __init__(self, ee_link, moveit_group):
        rospy.init_node('compliant_pose_controller', anonymous=True)

        # start position publisher
        self.joint_command_pub = rospy.Publisher("/joint_command", JointState, queue_size=1)
        rospy.Subscriber("joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("pose_command", PoseStamped, self.pose_command_callback)

        self.joint_state = None
        self.joint_names = ['shoulder_lift_joint', 'shoulder_pan_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joint_positions = np.zeros(6)
        self.joint_reorder = [2,1,0,3,4,5]
        self.prev_joint_command = None
        self.commands = Queue.Queue()

        self.lower_lims = (np.pi/180)*np.array([5.0, -120.0, 5.0, -150.0, -175.0, 95.0])
        self.upper_lims = (np.pi/180)*np.array([175.0, 5.0, 175.0, 5.0, 5.0, 265.0])
        self.max_joint_disp = 0.2

        # MoveIt
        self.group_name = moveit_group
        self.ee_link = ee_link

        self.ik_srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        self.ik_timeout = 10.0
        self.ik_attempts = 0
        self.avoid_collisions = False

        r = rospkg.RosPack()
        path = r.get_path('ur5e_compliant_controller')
        robot = URDF.from_xml_file(path+"/config/ur5e.urdf")
        self.kdl_kin = KDLKinematics(robot, "base_link", "ee_link")

    def joint_state_callback(self, data):
        self.joint_positions[self.joint_reorder] = data.position
        self.joint_state = data

    def pose_command_callback(self, data):
        self.commands.put(data)

    # Get IK using MoveIt
    def inverse_kinematics(self, pose):
        joint_constraints = list()
        for i in range(6):
            joint_constraints.append(JointConstraint(
                joint_name=self.joint_state.name[i],
                position=self.joint_state.position[i],
                tolerance_above=self.max_joint_disp,
                tolerance_below=self.max_joint_disp,
                weight=1.0
            ))

        req = GetPositionIKRequest()
        req.ik_request.group_name = self.group_name
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout = rospy.Duration(self.ik_timeout)
        req.ik_request.attempts = self.ik_attempts
        req.ik_request.avoid_collisions = self.avoid_collisions
        req.ik_request.robot_state = RobotState(joint_state=self.joint_state)
        req.ik_request.constraints = Constraints(joint_constraints=joint_constraints)

        try:
            resp = self.ik_srv.call(req)
            return resp.solution.joint_state
        except rospy.ServiceException as e:
            rospy.logerr('Service exception: ' + str(e))
            resp = GetPositionIKResponse()
            resp.error_code = 99999
            return resp

    def run(self):
        print('Waiting for joint state...')
        while self.joint_state is None:
            time.sleep(1)

        self.prev_joint_command = self.joint_state

        rate = rospy.Rate(1)
        print('Starting controller...')
        while not rospy.is_shutdown():
            # Wait for command
            if self.commands.empty():
                continue
            else:
                command = self.commands.get()

            current_joint_pos = copy.copy(self.joint_positions)
            target_joint_state = self.inverse_kinematics(command)
            target_joint_pos = np.array(list(target_joint_state.position))

            print(current_joint_pos)
            print(target_joint_pos)

            speed = 0.25
            max_disp = np.max(np.abs(target_joint_pos-current_joint_pos))
            end_time = max_disp / speed
            print(end_time)

            if max_disp > self.max_joint_disp:
                rospy.logerr('Requested movement is too large.')
                return

            traj = [InterpolatedUnivariateSpline([0.,end_time],[current_joint_pos[i], target_joint_pos[i]],k=1) for i in range(6)]
            traj_vel = InterpolatedUnivariateSpline([0.,end_time/2, end_time], [0, 0.1, 0],k=1)
            start_time, loop_time = time.time(), 0
            while loop_time < end_time:
                loop_time = time.time() - start_time
                joint_state = JointState(
                    position=[traj[j](loop_time) for j in range(6)],
                    velocity=[traj_vel(loop_time)] * 6,
                )
                self.joint_command_pub.publish(joint_state)

            joint_state = JointState(
                position=[traj[j](loop_time) for j in range(6)],
                velocity=[0] * 6
            )
            self.joint_command_pub.publish(joint_state)

            self.prev_joint_command = joint_state

            rate.sleep()

if __name__ == "__main__":
    ee_link = 'rg2_eef_link'
    moveit_group = 'manipulator'
    pose_controller = ur5e_position_controller(ee_link, moveit_group)
    pose_controller.run()
