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
            self.cmd.euler[0] = self.roll
            self.cmd.euler[1] = self.pitch
            self.cmd.euler[2] = self.yaw
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


## 此为脑控机器狗演示例程
import socket
import time
if __name__ == '__main__':
    dogbot = Unitree_Robot()

    # dogbot.robot_walking(gaitType=1, forwardSpeed=0.2, sidewaySpeed=0.0, rotateSpeed=0.0,
    #                                             speedLevel=0, bodyHeight=0.0)
    # time.sleep(2)
    # dogbot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0, rotateSpeed=0.0,
    #                      speedLevel=0, bodyHeight=0.0)

    # 遍历接收udp，执行运动
    while True:
        #接收指令
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.bind(('', 12345)) # 接收任意ip，对本机12345端口发起的请求
        data, addr = udp_socket.recvfrom(1024) # 缓冲区大小为 1024
        print("Received data from matlab(win):", data.decode())
        udp_socket.close()

        #发送指令
        ctrl_str = data.decode()

        print(ctrl_str)
        #
        # dogbot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0, rotateSpeed=0.2,
        #                                             speedLevel=0, bodyHeight=0.0)
        if ctrl_str == '1\\n': #forward
            dogbot.robot_walking(gaitType=1, forwardSpeed=0.2, sidewaySpeed=0.0, rotateSpeed=0.0,
                                                                             speedLevel=0, bodyHeight=0.0)
            # time.sleep(2)
            # dogbot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0, rotateSpeed=0.0,
            #                      speedLevel=0, bodyHeight=0.0)
        elif ctrl_str == '2\\n': #backward
            dogbot.robot_walking(gaitType=1, forwardSpeed=-0.2, sidewaySpeed=0.0, rotateSpeed=0.0,
                                 speedLevel=0, bodyHeight=0.0)
            # time.sleep(2)
            # dogbot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0, rotateSpeed=0.0,
            #                      speedLevel=0, bodyHeight=0.0)
        elif ctrl_str == '3\\n': #left
            dogbot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0, rotateSpeed=0.2,
                                 speedLevel=0, bodyHeight=0.0)
            # time.sleep(2)
            # dogbot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0, rotateSpeed=0.0,
            #                      speedLevel=0, bodyHeight=0.0)
        elif ctrl_str == '4\\n': #right
            dogbot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0, rotateSpeed=-0.2,
                                 speedLevel=0, bodyHeight=0.0)
            # time.sleep(2)
            # dogbot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0, rotateSpeed=0.0,
            #                      speedLevel=0, bodyHeight=0.0)
        else:
            pass

# 参数备份
# gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0,
#                       rotateSpeed=0.0, speedLevel=0, bodyHeight=0.0, footRaiseHeight=0.0




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
