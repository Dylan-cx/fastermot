#!/usr/bin/python

import sys
import time
import math
sys.path.append('/home/robot112/fastermot/src/unitree_legged_sdk-go1/lib/python/amd64')
# sys.path.append('../lib/python/amd64')
import robot_interface as sdk

HIGHLEVEL = 0xee
LOWLEVEL = 0xff
udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

class Unitree_Robot():
    # unitree_robot = robot_interface.RobotInterface()
    # robot_state = sdk.HighState()

    def __init__(self):
        self.HIGHLEVEL = HIGHLEVEL
        self.LOWLEVEL = LOWLEVEL

        # self.udp = sdk.UDP(self.HIGHLEVEL, 8080, "192.168.123.161", 8082)
        self.udp = udp

        self.cmd = sdk.HighCmd()
        self.state = sdk.HighState()
        self.udp.InitCmdData(self.cmd)

        self.cmd.mode = 0
        self.cmd.gaitType = 0
        self.cmd.speedLevel = 0
        self.cmd.footRaiseHeight = 0.0
        self.cmd.bodyHeight = 0.0

        self.cmd.euler = [0, 0, 0]
        self.cmd.velocity = [0.0, 0.0]
        self.cmd.yawSpeed = 0.0
        self.cmd.reserve = 0

        self.mode = 0
        self.gaitType = 0
        self.speedLevel = 0
        self.footRaiseHeight = 0.0
        self.forwardSpeed = 0.0
        self.sidewaySpeed = 0.0
        self.rotateSpeed = 0.0
        self.bodyHeight = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.quit_dance_time = 0

    def cmd_init(self):
        self.mode = 0
        self.gaitType = 0
        self.speedLevel = 0
        self.footRaiseHeight = 0.0
        self.forwardSpeed = 0.0
        self.sidewaySpeed = 0.0
        self.rotateSpeed = 0.0
        self.bodyHeight = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.quit_dance_time = 0

    def send_UDP(self):
        self.udp.Send()

    def recv_UDP(self):
        self.udp.Recv()

    def getState(self, state):
        self.udp.GetRecv(state)
        return state

    def robot_control(self):
        self.cmd.mode = 0
        self.cmd.gaitType = 0
        self.cmd.speedLevel = 0
        self.cmd.footRaiseHeight = 0.0
        self.cmd.bodyHeight = 0.0

        self.cmd.euler = [0, 0, 0]
        self.cmd.velocity = [0.0, 0.0]
        self.cmd.yawSpeed = 0.0
        self.cmd.reserve = 0

        if self.mode == 1:
            self.cmd.mode = self.mode
            # self.cmd.euler[0] = self.roll
            # self.cmd.euler[1] = self.pitch
            # self.cmd.euler[2] = self.yaw
            self.cmd.euler = [self.roll, self.pitch, self.yaw]
            self.cmd.bodyHeight = self.bodyHeight
        elif self.mode == 2:
            self.cmd.mode = self.mode
            self.cmd.gaitType = self.gaitType
            self.cmd.speedLevel = self.speedLevel
            self.cmd.footRaiseHeight = self.footRaiseHeight
            self.cmd.bodyHeight = self.bodyHeight
            # print(self.forwardSpeed)
            # print(self.cmd.velocity)
            # print(('__________'))
            self.cmd.velocity = [self.forwardSpeed, self.sidewaySpeed]
            # self.cmd.velocity[0] = self.forwardSpeed
            # self.cmd.velocity[1] = self.sidewaySpeed
            # print(self.cmd.velocity)
            self.cmd.yawSpeed = self.rotateSpeed
        else:
            self.cmd.mode = self.mode

        self.udp.SetSend(self.cmd)

    def robot_walking(self, gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0,
                      rotateSpeed=0.0, speedLevel=0, bodyHeight=0.0, footRaiseHeight=0.0):
        self.cmd_init()
        self.gaitType = gaitType
        self.speedLevel = speedLevel
        self.footRaiseHeight = footRaiseHeight

        self.forwardSpeed = forwardSpeed
        self.sidewaySpeed = sidewaySpeed
        self.rotateSpeed = rotateSpeed

        self.cmd.bodyHeight = bodyHeight
        self.mode = 2
        self.recv_UDP()
        robot_state = self.getState(self.state)
        self.robot_control()
        self.send_UDP()
        return robot_state

    def robot_pose(self, roll=0.0, pitch=0.0, yaw=0.0, bodyHeight=0.0):
        self.cmd_init()
        self.bodyHeight = bodyHeight# 趴下站立
        self.yaw = yaw# z左右
        self.pitch = pitch# x前后
        self.roll = roll# y横转
        self.mode = 1
        self.recv_UDP()
        robot_state = self.getState(self.state)
        self.robot_control()
        self.send_UDP()
        return robot_state

    def jump_yaw(self):# jump
        self.cmd_init()
        self.mode = 10
        self.recv_UDP()
        robot_state = self.getState(self.state)
        self.robot_control()
        self.send_UDP()
        return robot_state

    def straight_hand(self):# 拜年
        self.cmd_init()
        self.mode = 11
        self.recv_UDP()
        robot_state = self.getState(self.state)
        self.robot_control()
        self.send_UDP()
        return robot_state

    def robot_dance(self, dance_genre = 1):
        self.cmd_init()
        if (dance_genre == 1):
            self.mode = 12
        elif (dance_genre == 2):
            self.mode = 13
        self.recv_UDP()
        robot_state = self.getState(self.state)
        self.robot_control()
        self.send_UDP()
        return robot_state

    def quit_dance(self):
        self.quit_dance_time += 1
        if (self.quit_dance_time < 500):
            self.mode = 2
        if (self.quit_dance_time >= 500):
            self.mode = 1
        self.recv_UDP()
        robot_state = self.getState(self.state)
        self.robot_control()
        self.send_UDP()
        return robot_state


# if __name__ == '__main__':
#
#     HIGHLEVEL = 0xee
#     LOWLEVEL  = 0xff
#
#     udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
#
#     cmd = sdk.HighCmd()
#     state = sdk.HighState()
#     udp.InitCmdData(cmd)
#
#     motiontime = 0
#     while True:
#         time.sleep(0.002)
#         motiontime = motiontime + 1
#
#         udp.Recv()
#         udp.GetRecv(state)
#
#         # print(motiontime)
#         # print(state.imu.rpy[0])
#         # print(motiontime, state.motorState[0].q, state.motorState[1].q, state.motorState[2].q)
#         # print(state.imu.rpy[0])
#
#         cmd.mode = 0      # 0:idle, default stand      1:forced stand     2:walk continuously
#         cmd.gaitType = 0
#         cmd.speedLevel = 0
#         cmd.footRaiseHeight = 0
#         cmd.bodyHeight = 0
#         cmd.euler = [0, 0, 0]
#         cmd.velocity = [0, 0]
#         cmd.yawSpeed = 0.0
#         cmd.reserve = 0
#
#         # cmd.mode = 2
#         # cmd.gaitType = 1
#         # # cmd.position = [1, 0]
#         # # cmd.position[0] = 2
#         # cmd.velocity = [-0.2, 0] # -1  ~ +1
#         # cmd.yawSpeed = 0
#         # cmd.bodyHeight = 0.1
#
#         # if(motiontime > 0 and motiontime < 1000):
#         #     cmd.mode = 1
#         #     cmd.euler = [-0.3, 0, 0]
#         #
#         # if(motiontime > 1000 and motiontime < 2000):
#         #     cmd.mode = 1
#         #     cmd.euler = [0.3, 0, 0]
#         #
#         # if(motiontime > 2000 and motiontime < 3000):
#         #     cmd.mode = 1
#         #     cmd.euler = [0, -0.2, 0]
#         #
#         # if(motiontime > 3000 and motiontime < 4000):
#         #     cmd.mode = 1
#         #     cmd.euler = [0, 0.2, 0]
#         #
#         # if(motiontime > 4000 and motiontime < 5000):
#         #     cmd.mode = 1
#         #     cmd.euler = [0, 0, -0.2]
#         #
#         # if(motiontime > 5000 and motiontime < 6000):
#         #     cmd.mode = 1
#         #     cmd.euler = [0.2, 0, 0]
#         #
#         # if(motiontime > 6000 and motiontime < 7000):
#         #     cmd.mode = 1
#         #     cmd.bodyHeight = -0.2
#         #
#         # if(motiontime > 7000 and motiontime < 8000):
#         #     cmd.mode = 1
#         #     cmd.bodyHeight = 0.1
#         #
#         # if(motiontime > 8000 and motiontime < 9000):
#         #     cmd.mode = 1
#         #     cmd.bodyHeight = 0.0
#         #
#         # if(motiontime > 9000 and motiontime < 11000):
#         #     cmd.mode = 5
#         #
#         # if(motiontime > 11000 and motiontime < 13000):
#         #     cmd.mode = 6
#         #
#         # if(motiontime > 13000 and motiontime < 14000):
#         #     cmd.mode = 0
#         #
#         # if(motiontime > 14000 and motiontime < 18000):
#         #     cmd.mode = 2
#         #     cmd.gaitType = 2
#         #     cmd.velocity = [0.4, 0] # -1  ~ +1
#         #     cmd.yawSpeed = 2
#         #     cmd.footRaiseHeight = 0.1
#         #     # printf("walk\n")
#         #
#         # if(motiontime > 18000 and motiontime < 20000):
#         #     cmd.mode = 0
#         #     cmd.velocity = [0, 0]
#         #
#         # if(motiontime > 20000 and motiontime < 24000):
#         #     cmd.mode = 2
#         #     cmd.gaitType = 1
#         #     cmd.velocity = [0.2, 0] # -1  ~ +1
#         #     cmd.bodyHeight = 0.1
#         #     # printf("walk\n")
#
#
#         udp.SetSend(cmd)
#         udp.Send()
