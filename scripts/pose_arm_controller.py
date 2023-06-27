#!/usr/bin/env python3

import sys
import rospy
import rospkg
import numpy as np
from copy import deepcopy
import copy
import time
from queue import Queue
from scipy.interpolate import InterpolatedUnivariateSpline
import moveit_commander

from std_msgs.msg import Float64MultiArray, Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

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
        self.commands = Queue()

        self.lower_lims = (np.pi/180)*np.array([5.0, -120.0, 5.0, -150.0, -175.0, 95.0])
        self.upper_lims = (np.pi/180)*np.array([175.0, 5.0, 175.0, 5.0, 5.0, 265.0])
        self.max_joint_disp = np.array([0.2, 0.2, 0.2, 0.4, 0.4, 0.6])

        # MoveIt
        self.group_name = moveit_group
        self.ee_link = ee_link
        moveit_commander.roscpp_initialize(sys.argv)
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.move_group.set_planning_time = 0.1
        self.move_group.set_goal_position_tolerance(0.01)
        self.move_group.set_goal_orientation_tolerance(0.01)

    def joint_state_callback(self, data):
        self.joint_positions[self.joint_reorder] = data.position
        self.joint_state = data

    def pose_command_callback(self, data):
        self.commands.put(data)

    def run(self):
        print('Waiting for joint state...')
        while self.joint_state is None:
            time.sleep(1)

        rate = rospy.Rate(1)
        print('Starting controller...')
        while not rospy.is_shutdown():
            # Wait for command
            if self.commands.empty():
                continue
            else:
                command = self.commands.get()

            current_joint_pos = copy.copy(self.joint_positions)
            self.move_group.clear_pose_targets()
            self.move_group.set_pose_target(command)
            success, traj, planning_time, err = self.move_group.plan()
            if not success:
                rospy.logerr('MoveIt planning/IK failed...')
                continue

            target_joint_pos = np.array(list(traj.joint_trajectory.points[-1].positions))

            rospy.loginfo('Current: {}'.format(current_joint_pos))
            rospy.loginfo('Target: {}'.format(target_joint_pos))

            speed = 0.25
            joint_disp = np.abs(target_joint_pos-current_joint_pos)
            max_disp = np.max(joint_disp)
            end_time = max_disp / speed

            if np.any(np.array(joint_disp) > self.max_joint_disp):
                rospy.logerr('Requested movement is too large: {}.'.format(joint_disp))
                continue

            traj = [InterpolatedUnivariateSpline([0.,end_time],[current_joint_pos[i], target_joint_pos[i]],k=1) for i in range(6)]
            traj_vel = InterpolatedUnivariateSpline([0.,end_time/2, end_time], [0, 0.05, 0],k=1)
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

            rate.sleep()

if __name__ == "__main__":
    ee_link = 'rg2_eef_link'
    moveit_group = 'manipulator'
    pose_controller = ur5e_position_controller(ee_link, moveit_group)
    pose_controller.run()
