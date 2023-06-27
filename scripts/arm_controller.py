#! /usr/bin/env python
import rospy
import rospkg
import numpy as np
from copy import deepcopy
import time
from scipy.interpolate import InterpolatedUnivariateSpline

from filter import PythonBPF

from ur_kinematics.ur_kin_py import forward, forward_link
from kinematics import analytical_ik, nearest_ik_solution

from std_msgs.msg import Float64MultiArray, Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped
from ur_dashboard_msgs.msg import SafetyMode
from ur_dashboard_msgs.srv import IsProgramRunning, GetSafetyMode
from std_msgs.msg import Bool

# Import the module
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

r = rospkg.RosPack()
path = r.get_path('ur5e_compliant_controller')
robot = URDF.from_xml_file(path+"/config/ur5e.urdf")
dummy_arm = URDF.from_xml_file(path+"/config/dummy_arm.urdf")

joint_vel_lim = 1.0
sample_rate = 500
control_arm_saved_zero = np.array([0.51031649, 1.22624958, 3.31996918, 0.93126088, 3.1199832, 9.78404331])

two_pi = np.pi*2
gripper_collision_points =  np.array([[0.04, 0.0, -0.21, 1.0], #fingertip
                                      [0.05, 0.04, 0.09,  1.0],  #hydraulic outputs
                                      [0.05, -0.04, 0.09,  1.0]]).T

## TODO Move some of the params in to a configuration file
class ur5e_arm():
    '''Defines velocity based controller for ur5e arm for use in teleop project
    '''
    safety_mode = -1
    shutdown = False
    jogging = False
    enabled = True
    joint_reorder = [2,1,0,3,4,5]
    breaking_stop_time = 0.1 #when stoping safely, executes the stop in 0.1s Do not make large!

    #throws an error and stops the arm if there is a position discontinuity in the
    #encoder input freater than the specified threshold
    #with the current settings of 100hz sampling, 0.1 radiands corresponds to
    #~10 rps velocity, which is unlikely to happen unless the encoder input is wrong
    position_jump_error = 0.1
    # ains = np.array([10.0]*6) #works up to at least 20 on wrist 3
    joint_p_gains_varaible = np.array([5.0, 5.0, 5.0, 10.0, 10.0, 10.0]) #works up to at least 20 on wrist 3
    joint_ff_gains_varaible = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    #default_pos = (np.pi/180)*np.array([90.0, -120.0, 90.0, -77.0, -90.0, 180.0])
    default_pos = (np.pi/180)*np.array([76., -84., 90., -96., -90., 165.])
    robot_ref_pos = deepcopy(default_pos)
    saved_ref_pos = None
    daq_ref_pos = deepcopy(default_pos)

    lower_lims = (np.pi/180)*np.array([5.0, -120.0, 5.0, -150.0, -175.0, 95.0])
    upper_lims = (np.pi/180)*np.array([175.0, 5.0, 175.0, 5.0, 5.0, 265.0])
    conservative_lower_lims = (np.pi/180)*np.array([45.0, -100.0, 45.0, -135.0, -135.0, 135.0])
    conservative_upper_lims = (np.pi/180)*np.array([135, -45.0, 140.0, -45.0, -45.0, 225.0])
    # max_joint_speeds = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    max_joint_speeds = 3.0 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    max_joint_acc = 5.0 * np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
    homing_joint_speeds = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    jogging_joint_speeds = 2.0 * homing_joint_speeds
    homing_joint_acc = 2.0 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # max_joint_speeds = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])*0.1
    #default control arm setpoint - should be calibrated to be 1 to 1 with default_pos
    #the robot can use relative joint control, but this saved defailt state can
    #be used to return to a 1 to 1, absolute style control
    control_arm_def_config = np.mod(control_arm_saved_zero,np.pi*2)
    control_arm_ref_config = deepcopy(control_arm_def_config) #can be changed to allow relative motion

    #define fields that are updated by the subscriber callbacks
    current_joint_positions = np.zeros(6)
    current_joint_velocities = np.zeros(6)

    current_cmd_joint_positions = np.zeros(6)
    current_cmd_joint_velocities = np.zeros(6)

    #DEBUG
    current_daq_rel_positions = np.zeros(6)
    current_daq_rel_positions_waraped = np.zeros(6)

    first_daq_callback = True
    first_wrench_callback = True

    # define bandpass filter parameters
    fl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fh = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    fs = sample_rate
    filter = PythonBPF(fs, fl, fh)

    wrench = np.zeros(6)
    est_wrench_int_term = np.zeros(6)
    load_mass_publish_rate = 1
    load_mass_publish_index = 0

    ## control loop extra parameters
    current_wrench_global = np.zeros(6)
    current_joint_torque = np.zeros(6)

    joint_torque_error = np.zeros(6)
    wrench_global_error = np.zeros(6)

    joint_inertia = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    joint_stiffness = 350 * np.array([0.6, 1.0, 1.0, 1.0, 1.0, 1.0])
    zeta = 1.0

    recent_data_focus_coeff = 0.99
    p = 1 / recent_data_focus_coeff

    last_command_joint_velocities = np.zeros(6)

    tree = kdl_tree_from_urdf_model(robot)
    # print tree.getNrOfSegments()
    # forwawrd kinematics
    chain = tree.getChain("base_link", "wrist_3_link")
    # print chain.getNrOfJoints()
    kdl_kin = KDLKinematics(robot, "base_link", "wrist_3_link")

    tree_op = kdl_tree_from_urdf_model(dummy_arm)
    chain_op = tree_op.getChain("base_link", "wrist_3_link")
    kdl_kin_op = KDLKinematics(dummy_arm, "base_link", "wrist_3_link")

    def __init__(self, test_control_signal = False, conservative_joint_lims = False):
        '''set up controller class variables & parameters'''

        if conservative_joint_lims:
            self.lower_lims = self.conservative_lower_lims
            self.upper_lims = self.conservative_upper_lims

        #keepout (limmited to z axis height for now)
        self.keepout_enabled = True
        self.z_axis_lim = 0.0 # floor 0.095 #short table # #0.0 #table

        #launch nodes
        rospy.init_node('compliant_controller', anonymous=True)
        #start subscribers
        rospy.Subscriber("joint_command", JointState, self.joint_command_callback)
        rospy.Subscriber("reset_wrench", Bool, self.reset_wrench_callback)

        #start robot state subscriber (detects fault or estop press)
        rospy.Subscriber('/ur_hardware_interface/safety_mode',SafetyMode, self.safety_callback)
        #joint feedback subscriber
        rospy.Subscriber("joint_states", JointState, self.joint_state_callback)
        #wrench feedback
        rospy.Subscriber("wrench", WrenchStamped, self.wrench_callback)
        #service to check if robot program is running
        rospy.wait_for_service('/ur_hardware_interface/dashboard/program_running')
        self.remote_control_running = rospy.ServiceProxy('ur_hardware_interface/dashboard/program_running', IsProgramRunning)
        #service to check safety mode
        rospy.wait_for_service('/ur_hardware_interface/dashboard/get_safety_mode')
        self.safety_mode_proxy = rospy.ServiceProxy('/ur_hardware_interface/dashboard/get_safety_mode', GetSafetyMode)
        #start subscriber for deadman enable
        #rospy.Subscriber('/enable_move',Bool,self.enable_callback)
        #start subscriber for jogging enable

        #start vel publisher
        self.vel_pub = rospy.Publisher("/joint_group_vel_controller/command",
                            Float64MultiArray,
                            queue_size=1)

        #vertical direction force measurement publisher
        self.load_mass_pub = rospy.Publisher("/load_mass",
                            Float64MultiArray,
                            queue_size=1)
        self.test_data_pub = rospy.Publisher("/test_data",
                            Float64MultiArray,
                            queue_size=1)
        self.is_homing_pub = rospy.Publisher("/is_homing", Bool, queue_size=1)

        #ref pos publisher DEBUG
        self.daq_pos_pub = rospy.Publisher("/debug_ref_pos",
                            Float64MultiArray,
                            queue_size=1)
        self.daq_pos_wraped_pub = rospy.Publisher("/debug_ref_wraped_pos",
                            Float64MultiArray,
                            queue_size=1)
        self.ref_pos = Float64MultiArray(data=[0,0,0,0,0,0])
        #DEBUG
        # self.daq_pos_debug = Float64MultiArray(data=[0,0,0,0,0,0])
        # self.daq_pos_wraped_debug = Float64MultiArray(data=[0,0,0,0,0,0])

        #set shutdown safety behavior
        rospy.on_shutdown(self.shutdown_safe)
        time.sleep(0.5)
        self.stop_arm() #ensure arm is not moving if it was already

        self.velocity = Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.vel_ref = Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.load_mass = Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.test_data = Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.is_homing = False


        print("Joint Limmits: ")
        print(self.upper_lims)
        print(self.lower_lims)

        if not self.ready_to_move():
            print('User action needed before commands can be sent to the robot.')
            self.user_prompt_ready_to_move()
        else:
            print('Ready to move')

    def joint_state_callback(self, data):
        self.current_joint_state = deepcopy(data)
        self.current_joint_positions[self.joint_reorder] = data.position
        self.current_joint_velocities[self.joint_reorder] = data.velocity

    def reset_wrench_callback(self, data):
        self.first_wrench_callback = True

    def wrench_callback(self, data):
        self.current_wrench = np.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z, data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z])

        if self.first_wrench_callback:
            self.filter.calculate_initial_values(self.current_wrench)

        Ja = self.kdl_kin.jacobian(self.current_joint_positions)
        FK = self.kdl_kin.forward(self.current_joint_positions)
        p = FK[:3,3]
        R = FK[:3,:3]
        RT = np.transpose(R)

        Rp0 = np.cross(np.array(RT[:,0]).reshape(-1),np.array(p).reshape(-1))
        Rp1 = np.cross(np.array(RT[:,1]).reshape(-1),np.array(p).reshape(-1))
        Rp2 = np.cross(np.array(RT[:,2]).reshape(-1),np.array(p).reshape(-1))
        Rp = - np.array([Rp0, Rp1, Rp2])

        filtered_wrench = np.array(self.filter.filter(self.current_wrench))
	np.matmul(R, filtered_wrench[3:], out = self.current_wrench_global[3:])
        #np.matmul(RT, filtered_wrench[:3], out = self.current_wrench_global[3:])

        self.current_wrench_global[:3] = np.matmul(R, filtered_wrench[:3]) + np.matmul(Rp, filtered_wrench[3:])
        np.matmul(Ja.transpose(), self.current_wrench_global, out = self.current_joint_torque)
        self.first_wrench_callback = False

    def joint_command_callback(self, data):
        self.current_cmd_joint_positions[:] = data.position
        #self.current_cmd_joint_positions = np.mod(self.current_cmd_joint_positions+np.pi,two_pi)-np.pi
        self.current_cmd_joint_velocities[:] = data.velocity

    def safety_callback(self, data):
        '''Detect when safety stop is triggered'''
        self.safety_mode = data.mode
        if not data.mode == 1:
            #estop or protective stop triggered
            #send a breaking command
            print('\nFault Detected, sending stop command\n')
            self.stop_arm() #set commanded velocities to zero
            print('***Please clear the fault and restart the UR-Cap program before continuing***')

            #wait for user to fix the stop
            # self.user_wait_safety_stop()

    def enable_callback(self, data):
        '''Detects the software enable/disable safety switch'''
        self.enabled = data.data

    def user_wait_safety_stop(self):
        #wait for user to fix the stop
        while not self.safety_mode == 1:
            raw_input('Safety Stop or other stop condition enabled.\n Correct the fault, then hit enter to continue')

    def ensure_safety_mode(self):
        '''Blocks until the safety mode is 1 (normal)'''
        while not self.safety_mode == 1:
            raw_input('Robot safety mode is not normal, \ncheck the estop and correct any faults, then restart the external control program and hit enter. ')

    def get_safety_mode(self):
        '''Calls get safet mode service, does not return self.safety_mode, which is updated by the safety mode topic, but should be the same.'''
        return self.safety_mode_proxy().safety_mode.mode

    def ready_to_move(self):
        '''returns true if the safety mode is 1 (normal) and the remote program is running'''
        return self.get_safety_mode() == 1 and self.remote_control_running()

    def user_prompt_ready_to_move(self):
        '''Blocking dialog to get the user to reset the safety warnings and start the remote program'''
        while True:
            if not self.get_safety_mode() == 1:
                print(self.get_safety_mode())
                raw_input('Safety mode is not Normal. Please correct the fault, then hit enter.')
            else:
                break
        while True:
            if not self.remote_control_running():
                raw_input('The remote control URCap program has been pause or was not started, please restart it, then hit enter.')
            else:
                break
        print('\nRemote control program is running, and safety mode is Normal\n')

    def set_current_config_as_control_ref_config(self,
                                                 reset_robot_ref_config_to_current = True,
                                                 interactive = True):
        '''
        Initialize the encoder value of the dummy arm
        '''
        if interactive:
            _ = raw_input("Hit enter when ready to set the control arm ref pos.")
        self.control_arm_ref_config = np.mod(deepcopy(self.current_cmd_joint_positions),np.pi*2)
        if reset_robot_ref_config_to_current:
            self.robot_ref_pos = deepcopy(self.current_joint_positions)
        print("Control Arm Ref Position Setpoint:\n{}\n".format(self.control_arm_def_config))

    def is_joint_position(self, position):
        '''Verifies that this is a 1dim numpy array with len 6'''
        if isinstance(position, np.ndarray):
            return position.ndim==1 and len(position)==6
        else:
            return False

    def shutdown_safe(self):
        '''Should ensure that the arm is brought to a stop before exiting'''
        self.shutdown = True
        print('Stopping -> Shutting Down')
        self.stop_arm()
        print('Stopped')

    def stop_arm(self, safe = False):
        '''Commands zero velocity until sure the arm is stopped. If safe is False
        commands immediate stop, if set to a positive value, will stop gradually'''

        if safe:
            loop_rate = rospy.Rate(200)
            start_time = time.time()
            start_vel = deepcopy(self.current_joint_velocities)
            max_accel = np.abs(start_vel/self.breaking_stop_time)
            vel_mask = np.ones(6)
            vel_mask[start_vel < 0.0] = -1
            while np.any(np.abs(self.current_joint_velocities)>0.0001) and not rospy.is_shutdown():
                command_vels = [0.0]*6
                loop_time = time.time() - start_time
                for joint in range(len(command_vels)):
                    vel = start_vel[joint] - vel_mask[joint]*max_accel[joint]*loop_time
                    if vel * vel_mask[joint] < 0:
                        vel = 0
                    command_vels[joint] = vel
                self.vel_pub.publish(Float64MultiArray(data = command_vels))
                if np.sum(command_vels) == 0:
                    break
                loop_rate.sleep()

        while np.any(np.abs(self.current_joint_velocities)>0.0001):
            self.vel_pub.publish(Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def in_joint_lims(self, position):
        '''expects an array of joint positions'''
        return np.all(self.lower_lims < position) and np.all(self.upper_lims > position)

    def identify_joint_lim(self, position):
        '''expects an array of joint positions. Prints a human readable list of
        joints that exceed limmits, if any'''
        if self.in_joint_lims(position):
            print("All joints ok")
            return True
        else:
            for i, pos in enumerate(position):
                if pos<self.lower_lims[i]:
                    print('Joint {}: Position {:.5} exceeds lower bound {:.5}'.format(i,pos,self.lower_lims[i]))
                if pos>self.upper_lims[i]:
                    print('Joint {}: Position {:.5} exceeds upper bound {:.5}'.format(i,pos,self.lower_lims[i]))
            return False

    # homing() function: initialzing UR to default position
    def homing(self):
        print('Moving to home position...')

        rate = rospy.Rate(sample_rate)
        self.init_joint_admittance_controller()
        while not rospy.is_shutdown():
            self.joint_admittance_controller(ref_pos = self.default_pos,
                                             max_speeds = self.homing_joint_speeds,
                                             max_acc = self.homing_joint_acc)

            if not np.any(np.abs(self.default_pos - self.current_joint_positions)>0.02):
                print("Home position reached")
                break

            #wait
            rate.sleep()

        # Set current commanded positions to the current positions
        self.current_cmd_joint_positions = deepcopy(self.current_joint_positions)

        self.stop_arm(safe = True)
        self.vel_ref.data = np.array([0.0]*6)

    def return_collison_free_config(self, reference_positon):
        '''takes the proposed set of joint positions for the real robot and
        checks the forward kinematics for collisions with the floor plane and the
        defined gripper points. Returns the neares position with the same orientation
        that is not violating the floor constraint.'''
        pose = forward(reference_positon)
        collision_positions = np.dot(pose, gripper_collision_points)

        min_point = np.argmin(collision_positions[2,:])
        collision = collision_positions[2,min_point] < self.z_axis_lim
        if collision:
            # print('Z axis overrun: {}'.format(pose[2,3]))
            #saturate pose
            diff = pose[2,3] - collision_positions[2][min_point]
            # print(diff)
            pose[2,3] = self.z_axis_lim + diff
            # pose[2,3] = self.z_axis_lim
            #get joint ref
            reference_positon = nearest_ik_solution(analytical_ik(pose,self.upper_lims,self.lower_lims),self.current_joint_positions,threshold=0.2)
        return reference_positon

    def move(self, dialoge_enabled = True):
        ''' Main control loop '''
        rate = rospy.Rate(sample_rate)
        self.init_joint_admittance_controller(dialoge_enabled)

        while not self.shutdown and self.safety_mode == 1 and self.enabled: #shutdown is set on ctrl-c.
            self.joint_admittance_controller(ref_pos = self.current_cmd_joint_positions,
                                             max_speeds = self.max_joint_speeds,
                                             max_acc = self.max_joint_acc)

            #wait
            rate.sleep()
        self.stop_arm(safe = True)
        self.vel_ref.data = np.array([0.0]*6)

    # online torque error id
    def force_torque_error_estimation(self, position_error, force_error, force):
        if not np.any(np.abs(position_error)>0.01) and not np.any(np.abs(self.current_joint_velocities)>0.001):
            force_error += self.p / (1 + self.p) * (force - force_error)
            self.p = (self.p - self.p ** 2 / (1 + self.p)) / self.recent_data_focus_coeff
        return force_error

    # joint admittance controller initialization
    def init_joint_admittance_controller(self, dialoge_enabled=True):
        self.joint_torque_error = deepcopy(self.current_joint_torque)
        self.vel_admittance = np.zeros(6)

    # joint admittance controller
    def joint_admittance_controller(self, ref_pos, max_speeds, max_acc):
        np.clip(ref_pos, self.lower_lims, self.upper_lims, ref_pos)
        joint_pos_error = np.subtract(ref_pos, self.current_joint_positions)
        vel_ref_array = np.multiply(joint_pos_error, self.joint_p_gains_varaible)

        #self.joint_torque_error = self.force_torque_error_estimation(joint_pos_error, self.joint_torque_error, self.current_joint_torque)
        joint_torque_after_correction = self.current_joint_torque - self.joint_torque_error

        acc = (joint_torque_after_correction + self.joint_stiffness * joint_pos_error
                + 0.5 * 2 * self.zeta * np.sqrt(self.joint_stiffness * self.joint_inertia) * (self.current_cmd_joint_velocities - self.current_joint_velocities)
                - 0.5 * 2 * self.zeta * np.sqrt(self.joint_stiffness * self.joint_inertia) * self.current_joint_velocities) / self.joint_inertia
        np.clip(acc, -max_acc, max_acc, acc)
        self.vel_admittance += acc / sample_rate

        vel_ref_array[0] += self.vel_admittance[0]
        vel_ref_array[1] += self.vel_admittance[1]
        vel_ref_array[2] += self.vel_admittance[2]

        np.clip(vel_ref_array, self.last_command_joint_velocities - max_acc / sample_rate, self.last_command_joint_velocities + max_acc / sample_rate, vel_ref_array)
        np.clip(vel_ref_array, -max_speeds, max_speeds, vel_ref_array)
        self.last_command_joint_velocities = vel_ref_array

        #publish
        self.vel_ref.data = vel_ref_array
        self.vel_pub.publish(self.vel_ref)

    def run(self):
        ''' Run runs the move routine repeatedly '''
        while not rospy.is_shutdown():
            #start moving
            print('Starting Movement')
            self.move(dialoge_enabled = False)

if __name__ == "__main__":
    #This script is included for testing purposes
    print("Starting compliant arm controller")

    arm = ur5e_arm()
    time.sleep(1)
    arm.stop_arm()

    arm.homing()
    arm.run()

    arm.stop_arm()
