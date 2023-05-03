import rospy
import rospkg
import tf
import numpy as np
from copy import deepcopy

from std_msgs.msg import Float64MultiArray, Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

class FKPublisher(object):
    def __init__(self):
        rospy.init_node('fk_publisher', anonymous=True)

        self.ee_pose_pub = rospy.Publisher("/ee_pose", PoseStamped, queue_size=1)
        rospy.Subscriber("joint_states", JointState, self.joint_state_callback)
        self.joint_positions = np.zeros(6)
        self.joint_reorder = [2,1,0,3,4,5]

        r = rospkg.RosPack()
        path = r.get_path('ur5e_compliant_controller')
        robot = URDF.from_xml_file(path+"/config/ur5e.urdf")
        self.kdl_kin = KDLKinematics(robot, "base_link", "ee_link")

    def joint_state_callback(self, data):
        self.joint_positions[self.joint_reorder] = data.position

    def run(self):
        rate = rospy.Rate(500)
        while True:
            fk = self.kdl_kin.forward(self.joint_positions)
            pos = fk[:3,-1]
            rot = tf.transformations.quaternion_from_matrix(fk)

            ee_pose = PoseStamped(
                header=Header(frame_id='world'),
                pose=Pose(
                    position=Point(x=pos[0], y=pos[1], z=pos[2]),
                    orientation=Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
                )
            )
            self.ee_pose_pub.publish(ee_pose)

            rate.sleep()

if __name__ == '__main__':
    fk_pub = FKPublisher()
    fk_pub.run()
