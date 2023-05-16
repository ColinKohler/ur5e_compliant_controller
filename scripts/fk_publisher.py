#!/usr/bin/env python3

import rospy
import rospkg
import tf
import numpy as np
from copy import deepcopy

from std_msgs.msg import Float64MultiArray, Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

from moveit_msgs.srv import GetPositionFK
from moveit_msgs.srv import GetPositionFKRequest
from moveit_msgs.srv import GetPositionFKResponse

class FKPublisher(object):
  def __init__(self):
    rospy.init_node('fk_publisher', anonymous=True)

    self.ee_pose_pub = rospy.Publisher("/ee_pose", PoseStamped, queue_size=1)
    rospy.Subscriber("joint_states", JointState, self.joint_state_callback)

    self.joint_state = None
    self.fk_link = 'rg2_eef_link'
    self.fk_srv = rospy.ServiceProxy('/compute_fk', GetPositionFK)

  def joint_state_callback(self, data):
    self.joint_state = data

  def run(self):
    rate = rospy.Rate(100)
    print('Waiting for joint state...')
    while self.joint_state is None:
      rate.sleep()
      continue

    print('Starting FK service')
    while True:
      req = GetPositionFKRequest()
      req.header.frame_id = 'base_link'
      req.fk_link_names = [self.fk_link]
      req.robot_state.joint_state = self.joint_state
      try:
        resp = self.fk_srv.call(req)
      except rospy.ServiceException as e:
        rospy.logerr("Service exception: " + str(e))

      self.ee_pose_pub.publish(resp.pose_stamped[0])

    rate.sleep()

if __name__ == '__main__':
  fk_pub = FKPublisher()
  fk_pub.run()
