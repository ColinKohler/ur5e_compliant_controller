#! /usr/bin/env python
import rospy
import rospkg
import numpy as np
from copy import deepcopy
import time

from std_msgs.msg import Float64MultiArray, Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest, GetPositionFKResponse

class ur5e_position_controller(object):
    def __init__(self, ee_link, moveit_group):
        rospy.init_node('compliant_pose_controller', anonymous=True)

        # start position publisher
        self.ee_pose_pub = rospy.Publisher("/ee_pose", PoseStamped, queue_size=1)
        self.joint_command_pub = rospy.Publisher("/joint_command", JointState, queue_size=1)
        rospy.Subscriber("pose_command", PoseStamped, self.pose_command_callback)
        rospy.Subscriber("joint_states", JointState, self.joint_state_callback)

        # MoveIt
        self.group_name = moveit_group
        self.ee_link = ee_link

        self.ik_srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        self.ik_timeout = 1.0
        self.ik_attempts = 0
        self.avoid_collisions = False

        self.fk_srv = rospy.ServiceProxy('/compute_fk', GetPositionFK)

        self.current_joint_state = None

        self.max_pos_delta = 0.05

    def joint_state_callback(self, data):
        self.current_joint_state = deepcopy(data)

    def checkPoseDistance(self, pose_1, pose_2):
        pos_1 = np.array([pose_1.pose.position.x, pose_1.pose.position.y, pose_1.pose.position.z])
        pos_2 = np.array([pose_2.pose.position.x, pose_2.pose.position.y, pose_2.pose.position.z])

        return (np.linalg.norm(pos_1 - pos_2) < self.max_pos_delta)

    def pose_command_callback(self, data):
        print('pose cmd callback')
        if not self.checkPoseDistance(data, self.current_ee_pose):
          rospy.logerr('Requested movement is too large.')
          return

        cmd_joint_state = self.inverse_kinematics(data)
        cmd_joint_state.velocity = [0.025] * 6
        self.joint_command_pub.publish(cmd_joint_state)

    # Get IK using MoveIt
    def inverse_kinematics(self, pose):
        req = GetPositionIKRequest()
        req.ik_request.group_name = self.group_name
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout = rospy.Duration(self.ik_timeout)
        req.ik_request.attempts = self.ik_attempts
        req.ik_request.avoid_collisions = self.avoid_collisions

        try:
            resp = self.ik_srv.call(req)
            return resp.solution.joint_state
        except rospy.ServiceException as e:
            rospy.logerr('Service exception: ' + str(e))
            resp = GetPositionIKResponse()
            resp.error_code = 99999
            return resp

    # Get FK using MoveIt
    def forward_kinematics(self, joint_state):
        req = GetPositionFKRequest()
        req.header.frame_id = 'world'
        req.fk_link_names = [self.ee_link]
        req.robot_state.joint_state = joint_state

        try:
            resp = self.fk_srv.call(req)
            return resp.pose_stamped[0]
        except rospy.ServiceException as e:
            rospy.logerr('Service exception: ' + str(e))
            resp = GetPositionFKResponse()
            resp.error_code = 99999
            return resp

    def run(self):
        print('Waiting for joint state...')
        while self.current_joint_state is None:
            time.sleep(1)

        print('Starting controller...')
        while not rospy.is_shutdown():
            self.current_ee_pose = self.forward_kinematics(self.current_joint_state)
            self.ee_pose_pub.publish(self.current_ee_pose)
            time.sleep(2)

if __name__ == "__main__":
    ee_link = 'ee_link'
    moveit_group = 'ur5e_arm'
    pose_controller = ur5e_position_controller(ee_link, moveit_group)
    pose_controller.run()
