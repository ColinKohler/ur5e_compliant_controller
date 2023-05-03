import rospy
from geometry_msgs.msg import PoseStamped

import time
import copy

current_ee_pose = None

def ee_pose_callback(data):
  global current_ee_pose
  current_ee_pose = data

def main():
  global current_ee_pose
  rospy.init_node('test_pos_controller')

  sample_rate = 500
  ee_pose_sub = rospy.Subscriber('/ee_pose', PoseStamped, ee_pose_callback)
  ee_pose_pub = rospy.Publisher('/pose_command', PoseStamped, queue_size=1)

  print('Waiting for end effector pose...')
  rate = rospy.Rate(sample_rate)
  while current_ee_pose is None:
    rate.sleep()
  print('Current end effector pose: {}'.format(current_ee_pose.pose))

  while True:
    d = input()

    target_pose = copy.copy(current_ee_pose)
    target_pose.pose.position.z = current_ee_pose.pose.position.z - 0.025
    print('Target end effector pose: {}'.format(target_pose.pose))
  
    print('Sending target pose to compliant controller')
    ee_pose_pub.publish(target_pose)

#  rospy.spin()

if __name__ == '__main__':
  main()
