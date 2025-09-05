#!/usr/bin/python

import sys
import time
import torch
import math

sys.path.append("/src/unitree_legged_sdk-go1/example_py")
import Unitree_Robot as Unitree_Python_sdk
## 四足函数调用
unitree_robot = Unitree_Python_sdk.Unitree_Robot()

if __name__ == '__main__':
    motiontime = 0
    while True:
        motiontime += 1

        if (motiontime > 0 and motiontime < 1000):
            # 持续运动mode = 2
            # Uncomment the following line to control the pose of robot
            # The four arguments are: roll, pitch, yaw, bodyHeight.

            # state = unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0, rotateSpeed=0.0,
            #                                     speedLevel=0, bodyHeight=0.0)
            print(motiontime)

        if (motiontime >= 1000 and motiontime < 2000):
            # dance_genre = 1 或 2 大概是两种舞蹈？(mode = 12 or 13)

            # state = unitree_robot.robot_dance(dance_genre=1)
            print(motiontime)
        if (motiontime >= 2000 and motiontime < 3000):
            # 退出舞蹈模式(令mode=1 or 2)

            # state = unitree_robot.quit_dance()
            print(motiontime)

        if (motiontime >= 3000 and motiontime < 4000):
            # 控制姿态mode = 1
            # Uncomment the following line to control the movement of robot
            # The arguments are: gait type, forward speed, sideway speed, rotate speed, speed level and body height.
            # Gait type: 0.idle  1.trot  2.trot running  3.climb stair
            # Forward speed: unit: m/s -1.0 ~ 1.0
            # Sideway speed: unit: m/s -1.0 ~ 1.0
            # Rotate speed: unit: rad/s -1.0 ~ 1.0
            # Speed level: 0. default low speed. 1. medium speed
            # Body height: unit: m, default: 0.28m

            # state = unitree_robot.robot_pose(roll=0.0, pitch=0.0, yaw=-0.0, bodyHeight=-0.2)
            print(motiontime)
        if (motiontime >= 4000 and motiontime < 5000):
            # state = unitree_robot.robot_pose(roll=0.0, pitch=0.0, yaw=0.0, bodyHeight=0.0)
            print(motiontime)

        if (motiontime >= 5000 and motiontime < 6000):
            # jump(turning left) mode = 10

            # state = unitree_robot.jump_yaw()
            print(motiontime)

        if (motiontime >= 5000 and motiontime < 6000):
            # 拜年 mode = 11

            # state = unitree_robot.straight_hand()
            print(motiontime)

        if  (motiontime >= 7000 and motiontime < 8000):
            # return state of robot
            # imu                       //rpy[0], rpy[1], rpy[3]
            # gaitType                  // 0.idle  1.trot  2.trot running  3.climb stair
            # footRaiseHeight           // (unit: m, default: 0.08m), foot up height while walking
            # position                  // (unit: m), from own odometry in inertial frame, usually drift
            # bodyHeight                // (unit: m, default: 0.28m),
            # velocity                  // (unit: m/s), forwardSpeed, sideSpeed, rotateSpeed in body frame
            # yawSpeed                  // (unit: rad/s), rotateSpeed in body frame
            # footPosition2Body         // foot position relative to body
            # footSpeed2Body            // foot speed relative to body
            # footForce

            state0 = unitree_robot.state
            state = unitree_robot.getState(state0)
            print(motiontime, state.velocity[0])
            # print(state.imu.rpy[0])
