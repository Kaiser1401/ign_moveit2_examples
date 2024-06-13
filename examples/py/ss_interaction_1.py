#!/usr/bin/env python3
import time
from copy import deepcopy
from threading import Thread

import rclpy

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from pymoveit2 import MoveIt2, MoveIt2Gripper
from pymoveit2.robots import panda
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile

import subprocess
import math

import os

joint_configuration_default = [
    0.0,
    -0.7853981633974483,
    0.0,
    -2.356194490192345,
    0.0,
    1.5707963267948966,
    0.7853981633974483,
]

class SSInteraction(Node):
    def __init__(self):
        super().__init__("ss_interaction_1")

        # Create callback group that allows execution of callbacks in parallel without restrictions
        self._callback_group = ReentrantCallbackGroup()

        self._got_pose = False
        self._initial_object_pose:Pose = None

        # Create MoveIt 2 interface
        self._moveit2 = MoveIt2(
            node=self,
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(),
            end_effector_name=panda.end_effector_name(),
            group_name=panda.MOVE_GROUP_ARM,
            callback_group=self._callback_group,
        )

        # Use upper joint velocity and acceleration limits
        self._moveit2.max_velocity = 1.0
        self._moveit2.max_acceleration = 1.0

        # Create MoveIt 2 interface for gripper
        self._moveit2_gripper = MoveIt2Gripper(
            node=self,
            gripper_joint_names=panda.gripper_joint_names(),
            open_gripper_joint_positions=panda.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=panda.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=panda.MOVE_GROUP_GRIPPER,
            callback_group=self._callback_group,
        )

        # Use upper joint velocity and acceleration limits
        self._moveit2_gripper.max_velocity = 1.0
        self._moveit2_gripper.max_acceleration = 1.0

        # TODO: These? why? others? what would have been the default?
        #self._moveit2.planner_id = "RRTstarkConfigDefault"
        #self._moveit2_gripper.planner_id = "LBKPIECEkConfigDefault"

        # Create a subscriber for target pose
        self.__previous_target_pose = Pose()
        self.create_subscription(
            msg_type=PoseStamped,
            topic="/cube_pose_is",
            callback=self.target_pose_callback,
            qos_profile=QoSProfile(depth=1),
            callback_group=self._callback_group,
        )

        self.get_logger().info("Initialization successful.")

    def reset_object_pose(self, world, name, pose:Pose):
        # gz_moveit2_manipulation_1
        # interaction_cube

        cmd = "gz"
        params = ["service",
                  f"-s /world/{world}/set_pose",
                  "--reqtype gz.msgs.Pose",
                  "--reptype gz.msgs.Boolean",
                  "--timeout 500",
                  f"--req 'name:\"{name}\", position: {{x: '{pose.position.x:.2f}',y: '{pose.position.y:.2f}',z: '{pose.position.z:.2f}',}}, orientation: {{w: '{pose.orientation.w:.2f}', x: '{pose.orientation.x:.2f}', y: '{pose.orientation.y:.2f}', z: '{pose.orientation.z:.2f}',}}'"
                  ]
        print(params)
        self.get_logger().info(f"Resetting object '{name}' pose")
        #subprocess.run([cmd] + params)
        os.system(" ".join([cmd]+params))


    def target_pose_callback(self, msg: PoseStamped):
        """
        Plan and execute trajectory each time the target pose is changed
        """

        # Return if target pose is unchanged
        if self._got_pose and (msg.pose == self.__previous_target_pose):
            return

        if not self._initial_object_pose:
            #first time
            self._initial_object_pose = deepcopy(msg.pose)
        self._got_pose = True

#        self.get_logger().info("Target pose has changed.")

        # Update for next callback
        self.__previous_target_pose = msg.pose



    def exec_behaviour_1(self, loop_count=1):

        # Move_to_default
        self._moveit2.move_to_configuration(joint_configuration_default)
        self._moveit2.wait_until_executed()

        self._moveit2_gripper.open()
        self._moveit2_gripper.wait_until_executed()

        self._moveit2_gripper.close()
        self._moveit2_gripper.wait_until_executed()

        self.get_logger().info("Starting behaviour Sequence ...")

        for x in range(loop_count):

            self.get_logger().info(f"Sequence Loop {x+1}/{loop_count} ...")

            #randomize pose in gz ?

            while (not self._got_pose):
                self.get_logger().warn("No target pose yet... ", throttle_duration_sec=1)
                time.sleep(0.1)

            #get pose from gz
            pose_is_pre = self.__previous_target_pose

            # print(pose_is_pre)

            #emulte perception
            pose_hat, uncertainties = self.sample_around_pose(pose_is_pre)
            pose_dest = deepcopy(pose_hat)

            # set target
            pose_dest.position.x += 0.2
            pose_dest.position.y -= 0.1

            # predict from uncertainties

            #interact
            self.grip_and_place(pose_hat, pose_dest, bool_lift=False)

            #get pose from gz
            pose_is_post = self.__previous_target_pose

            # print(pose_is_post)

            #check success
            # TODO define success

            # update / learn with uncertainties and success

            # reset
            self.reset_object_pose("gz_moveit2_manipulation_1", "interaction_cube", self._initial_object_pose)
            self._got_pose = False

        self.get_logger().info(f"Sequence Done ({loop_count} Loops)")



    def sample_around_pose(self,pose_is: Pose) -> [Pose, ]:
        # TODO sample around the pose with some parameters to simulate perception pipeline

        #pose_sampled = PoseStamped()
        pose_sampled = deepcopy(pose_is)
        uncertainties = None
        return pose_sampled, uncertainties


    def grip_and_place(self, start:Pose, goal:Pose, bool_lift = True):

        # Open gripper
        self._moveit2_gripper.open()
        self._moveit2_gripper.wait_until_executed()

        # Move_to_default
        self._moveit2.move_to_configuration(joint_configuration_default)
        self._moveit2.wait_until_executed()

        above_s = deepcopy(start)
        above_s.position.z +=0.10

        above_g = deepcopy(goal)
        above_g.position.z +=0.10

        # to object  (over, down, grip, over)
        self._moveit2.move_to_pose(
            position=above_s.position,
            quat_xyzw=above_s.orientation,
        )
        self._moveit2.wait_until_executed()

        self._moveit2.move_to_pose(
            position=start.position,
            quat_xyzw=start.orientation,
        )
        self._moveit2.wait_until_executed()

        # close gripper
        self._moveit2_gripper.close()
        self._moveit2_gripper.wait_until_executed()

        if (bool_lift):
            self._moveit2.move_to_pose(
                position=above_s.position,
                quat_xyzw=above_s.orientation,
            )
            self._moveit2.wait_until_executed()

        # to object  (over, down, loose, over)
        if (bool_lift):
            self._moveit2.move_to_pose(
                position=above_g.position,
                quat_xyzw=above_g.orientation,
            )
            self._moveit2.wait_until_executed()

        self._moveit2.move_to_pose(
            position=goal.position,
            quat_xyzw=goal.orientation,
        )
        self._moveit2.wait_until_executed()

        # close gripper
        self._moveit2_gripper.open()
        self._moveit2_gripper.wait_until_executed()

        self._moveit2.move_to_pose(
            position=above_g.position,
            quat_xyzw=above_g.orientation,
        )
        self._moveit2.wait_until_executed()

        # Move_to_default
        self._moveit2.move_to_configuration(joint_configuration_default)
        self._moveit2.wait_until_executed()



def main(args=None):
    rclpy.init(args=args)

    ss_interaction = SSInteraction()

    # Spin the node in background thread(s)
    executor = rclpy.executors.MultiThreadedExecutor(3)
    executor.add_node(ss_interaction)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()

    # Wait for everything to setup
    sleep_duration_s = 2.0
    if rclpy.ok():
        ss_interaction.create_rate(1 / sleep_duration_s).sleep()


    ss_interaction.exec_behaviour_1(5)

    rclpy.shutdown()
    exit(0)



if __name__ == "__main__":
    main()

