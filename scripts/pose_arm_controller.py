#! /usr/bin/env python
import rospy
import rospkg
import numpy as np
from copy import deepcopy
import copy
import time
import Queue
from scipy.interpolate import InterpolatedUnivariateSpline

from std_msgs.msg import Float64MultiArray, Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped
from moveit_msgs.msg import RobotState, Constraints, JointConstraint
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest, GetPositionFKResponse

class ur5e_position_controller(object):
    def __init__(self, ee_link, moveit_group):
        rospy.init_node('compliant_pose_controller', anonymous=True)

        # start position publisher
        self.joint_command_pub = rospy.Publisher("/joint_command", JointState, queue_size=1)
        rospy.Subscriber("ee_pose", PoseStamped, self.ee_pose_callback)
        rospy.Subscriber("joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("pose_command", PoseStamped, self.pose_command_callback)

        self.joint_positions = np.zeros(6)
        self.joint_reorder = [2,1,0,3,4,5]
        self.ee_pose = None
        self.prev_joint_command = None
        self.commands = Queue.Queue()

        self.lower_lims = (np.pi/180)*np.array([5.0, -120.0, 5.0, -150.0, -175.0, 95.0])
        self.upper_lims = (np.pi/180)*np.array([175.0, 5.0, 175.0, 5.0, 5.0, 265.0])

        # MoveIt
        self.group_name = moveit_group
        self.ee_link = ee_link

        self.ik_srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        self.ik_timeout = 10.0
        self.ik_attempts = 0
        self.avoid_collisions = False

        self.max_pos_delta = 0.05

    def ee_pose_callback(self, data):
        self.ee_pose = deepcopy(data)

    def joint_state_callback(self, data):
        self.joint_state = data
        self.joint_positions[self.joint_reorder] = data.position

    def checkPoseDistance(self, pose_1, pose_2):
        pos_1 = np.array([pose_1.pose.position.x, pose_1.pose.position.y, pose_1.pose.position.z])
        pos_2 = np.array([pose_2.pose.position.x, pose_2.pose.position.y, pose_2.pose.position.z])

        print(np.linalg.norm(pos_1 - pos_2))

        return (np.linalg.norm(pos_1 - pos_2) < self.max_pos_delta)

    def pose_command_callback(self, data):
        print('pose cmd callback')
        self.commands.put(data)

    # Get IK using MoveIt
    def inverse_kinematics(self, pose):
        joint_constraints = list()
        for i in range(6):
            joint_constraints.append(JointConstraint(
                joint_name=self.joint_state.name[i],
                position=self.joint_state.position[i],
                tolerance_above=0.1,
                tolerance_below=0.1,
            ))

        req = GetPositionIKRequest()
        req.ik_request.group_name = self.group_name
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout = rospy.Duration(self.ik_timeout)
        req.ik_request.attempts = self.ik_attempts
        req.ik_request.avoid_collisions = self.avoid_collisions
        req.ik_request.robot_state = RobotState(joint_state=self.prev_joint_command)
        req.ik_request.constraints = Constraints(joint_constraints=joint_constraints)

        try:
            resp = self.ik_srv.call(req)
            print(resp)
            return resp.solution.joint_state
        except rospy.ServiceException as e:
            rospy.logerr('Service exception: ' + str(e))
            resp = GetPositionIKResponse()
            resp.error_code = 99999
            return resp

    def run(self):
        print('Waiting for joint state...')
        while self.ee_pose is None:
            time.sleep(1)

        self.prev_joint_command = self.joint_state

        rate = rospy.Rate(1)
        print('Starting controller...')
        while not rospy.is_shutdown():
            # Wait for command
            if self.commands.empty():
                continue

            #if not self.checkPoseDistance(data, self.ee_pose):
            #    rospy.logerr('Requested movement is too large.')
            #    return
            current_joint_pos = copy.copy(self.joint_positions)
            target_joint_state = self.inverse_kinematics(self.commands.get())
            target_joint_state.velocity = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02]
            target_joint_pos = np.array(list(target_joint_state.position)) * [1, -1, 1, -1, -1, 1]

            speed = 0.5
            max_disp = np.max(np.abs(target_joint_pos, current_joint_pos))
            end_time = max_disp / speed
            print(current_joint_pos)
            print(target_joint_pos)

            traj = [InterpolatedUnivariateSpline([0.,end_time],[current_joint_pos[i], target_joint_pos[i]],k=1) for i in range(6)]
            start_time, loop_time = time.time(), 0
            commands = list()
            while loop_time < end_time:
                loop_time = time.time() - start_time
                joint_state = JointState(
                    position=[traj[j](loop_time) for j in range(6)],
                    velocity=target_joint_state.velocity
                )
                commands.append(joint_state.position)
                #self.joint_command_pub.publish(joint_state)
            commands = np.array(commands)
            #import pdb; pdb.set_trace()
            #import matplotlib.pyplot as plt
            #for i in range(6):
            #    plt.plot(commands[:,i])
            #    plt.show()
            self.prev_joint_command = joint_state

            rate.sleep()

if __name__ == "__main__":
    ee_link = 'ee_link'
    moveit_group = 'ur5e_arm'
    pose_controller = ur5e_position_controller(ee_link, moveit_group)
    pose_controller.run()
