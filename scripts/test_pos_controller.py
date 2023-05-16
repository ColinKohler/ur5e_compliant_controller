#!/usr/bin/env python

import rospy
import time
import copy
import argparse

from geometry_msgs.msg import PoseStamped

current_ee_pose = None

def ee_pose_callback(data):
  global current_ee_pose
  current_ee_pose = data

def main(keyboard_control):
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

  if keyboard_control:
    while True:
      direction = input('Movement direction:')
      target_pose = copy.copy(current_ee_pose)

      if direction == 'q':
        target_pose.pose.position.x = current_ee_pose.pose.position.x + 0.025
      elif direction == 'e':
        target_pose.pose.position.x = current_ee_pose.pose.position.x - 0.025
      elif direction == 'd':
        target_pose.pose.position.y = current_ee_pose.pose.position.y + 0.025
      elif direction == 'a':
        target_pose.pose.position.y = current_ee_pose.pose.position.y - 0.025
      elif direction == 'w':
        target_pose.pose.position.z = current_ee_pose.pose.position.z + 0.025
      elif direction == 's':
        target_pose.pose.position.z = current_ee_pose.pose.position.z - 0.025
      else:
        print('Invalid command')
        continue

      print('Target end effector pose: {}'.format(target_pose.pose))
      print('Sending target pose to compliant controller')
      ee_pose_pub.publish(target_pose)
  else:
    poses = list()
    for i in range(1,6):
      pose = copy.deepcopy(current_ee_pose)
      pose.pose.position.y = current_ee_pose.pose.position.y + (i * 0.025)
      poses.append(pose)

    print('Running predefined traj')
    for pose in poses:
      ee_pose_pub.publish(pose)
      time.sleep(0.1)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--keyboard',
    action='store_true',
    default=False
  )
  args = parser.parse_args()

  main(args.keyboard)
