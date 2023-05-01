import rospy
from geometry_msgs.msg import PoseStamped

import time
import copy

current_ee_pose = None
def ee_pose_callback(data):
  current_ee_pose = data

if __name__ == '__main__':
  ee_pose_sub = rospy.Subscriber('compliant_controller/ee_pose', PoseStamped, ee_pose_callback)
  ee_pose_pub = rospy.Publisher('compliant_controller/pose_command', PoseStamped, queue_size=1)

  print('Waiting for end effector pose...')
  while current_ee_pose is None:
    time.sleep(1)
  print('Current end effector pose: {}'.format(current_ee_pose.pose))

  target_pose = copy.copy(current_ee_pose)
  target_pose.pose.position.z = current_ee_pose.pose.position.z - 0.025
  print('Target end effector pose: {}'.format(target_pose.pose))

  print('Sending target pose to compliant controller')
  ee_pose_pub.publish(target_pose)
  while True:
    time.sleep(1)
