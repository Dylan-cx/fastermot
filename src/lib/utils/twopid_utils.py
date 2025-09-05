from utils.PID_controller import PID_posi_2, PID_inc
import numpy as np
import math
import sys
#sys.path.append("/home/robot112/FairMOT-master/src/unitree_legged_sdk-go1/example_py")
#import Unitree_Python_sdk_go1 as Unitree_Python_sdk
sys.path.append("/home/robot112/fastermot/src/unitree_legged_sdk-go1/example_py")
import Unitree_Robot as Unitree_Python_sdk
## 四足函数调用
unitree_robot = Unitree_Python_sdk.Unitree_Robot()

## 位置式
PID = PID_posi_2(k=[0.45, 0.00, 0.0001], target=2, upper=1, lower=-1)
PIDw = PID_posi_2(k=[1, 0.0000, 0.00], target=0, upper=1, lower=-1)
## 增量式
# PID = PID_inc(k=[2.5, 0.175, 30], target=0, upper=np.pi/6, lower=-np.pi/6)

# TODO:如何区分左右转？(√）
#  TODO：误差e是否应该是负数？(√)
# TODO:PID内部的设计,输出合理吗？
# TODO：前后控制的2和3哪个是阈值？   改为2  (√）
# TODO：优化前进后退，缩小用于p控制的系数k？
#目标物体三位坐标
def xyz_true(target_xy_pixel, aligned_depth_frame, color_intrin_part):
    print('*' * 50)
    # 提取ppx,ppy,fx,fy
    ppx = color_intrin_part[0]
    ppy = color_intrin_part[1]
    fx = color_intrin_part[2]
    fy = color_intrin_part[3]
    #中心点坐标
    target_xy_pixel = target_xy_pixel
    target_depth = 0
    R_speed = 0
    F_speed = 0
    L_speed = 0
    B_speed = 0
    #加的480 是y变成左下角坐标
    if int(target_xy_pixel[0]) < 640 and int(target_xy_pixel[1]+479) < 480 and int(target_xy_pixel[1]+479) >= 0:
        target_depth = aligned_depth_frame.get_distance(int(target_xy_pixel[0]), int(target_xy_pixel[1]+479))
    if target_depth:
        # target_depth = 100
        target_xy_true = [(target_xy_pixel[0] - ppx) * target_depth / fx,
                          (target_xy_pixel[1]+479 - ppy) * target_depth / fy]
        print('中心点像素坐标：({}, {}) 实际坐标(mm)：（{:.3f}，{:.3f}） 深度(mm)：{:.3f}'.format(
                                                                                        target_xy_pixel[0],
                                                                                        target_xy_pixel[1] + 479,
                                                                                        target_xy_true[0] * 1000,
                                                                                        -target_xy_true[1] * 1000,
                                                                                        target_depth * 1000))



        # # 当前的半径
        # R = math.sqrt(math.pow(target_xy_true[0], 2) + math.pow(-target_xy_true[1], 2))
        # #取最大深度距离为10
        # #三维坐标下最大的x 和 最大的y，最远的点的坐标
        # target_xy_true_x_max = (639 - ppx) * 10 / fx
        # target_xy_true_y_max = (479 - ppy) * 10 / fy

        # 重新编写计算代码
        K_R_L = 1 / 320


        # #最大的半径，用来计算左转速度和距离函数比例系数
        # K_R_max = math.sqrt(math.pow(target_xy_true_x_max / 2,2) + math.pow(target_xy_true_y_max,2))
        # #左转右转函数比例系数（后续加正负号）
        # K_R_L = 1 / K_R_max
        #前进后退函数比例系数（后续加正负号）
        #中心距离
        Math_cm = 3
        #最远距离
        Max_cm = 7
        K_B = 1 / Math_cm
        K_F = 1 / (Max_cm - Math_cm)
        # print(target_xy_true_x_max)
        if target_depth > 2:
            if target_xy_true[0] > 0:
            #if target_xy_true[0] > (target_xy_true_x_max / 2):
                # 右转速度,(target_xy_pixel[0] - 320) / K_R_L大于0，且属于（0,1），是否合理？
                # PID的输出正负？
                # R_speed应该是正的，L_speed应该是负数
                R_speed = PIDw.cal_output((target_xy_pixel[0] - 320) * K_R_L)
                # R_speed = target_xy_pixel[0] * K_R_L - 1
                #前进速度
                # F_speed = (target_depth-Math_cm) * K_F
                # F_speed = float(target_depth) - 1
                #F_speed = target_depth - 3
                # F_speed应该是正的，B_speed应该是负数
                # F_speed = float(target_depth - 2) / 3
                F_speed = PID.cal_output(target_depth)
                if F_speed > 1:
                    F_speed = 1
                print('右转前进:右转速度:{},前进速度{}'.format(R_speed,F_speed))
                state = unitree_robot.robot_walking(gaitType=1, forwardSpeed=F_speed, sidewaySpeed=0.0, rotateSpeed=-R_speed,
                                                speedLevel=0, bodyHeight=0.0)
            elif target_xy_true[0] < 0:
            #elif target_xy_true[0] < (target_xy_true_x_max / 2):
                # 左转速度
                L_speed = PIDw.cal_output((target_xy_pixel[0] - 320) * K_R_L)
                # L_speed = target_xy_pixel[0] * K_R_L - 1
                # 前进速度
                # F_speed = (target_depth-Math_cm) * K_F
                # F_speed = float(target_depth) - 1
                #F_speed = target_depth - 3
                # F_speed = float(target_depth - 2) / 3
                F_speed = PID.cal_output(target_depth)
                if F_speed > 1:
                    F_speed = 1
                print('左转前进:左转速度:{},前进速度{}'.format(L_speed, F_speed))
                state = unitree_robot.robot_walking(gaitType=1, forwardSpeed=F_speed, sidewaySpeed=0.0, rotateSpeed=-L_speed,
                                                speedLevel=0, bodyHeight=0.0)
            else:
                #归零 静止不动
                R_speed = 0
                #F_speed = 0
                L_speed = 0
                #B_speed = 0
                print('静止不动:右转速度:' + R_speed + '左转速度:' + L_speed + '前进速度:' + F_speed + '后退速度:' + B_speed)
                state = unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0,rotateSpeed=0.0,
                                                    speedLevel=0, bodyHeight=0.0)
        else:
            if target_xy_true[0] > 0:
            #if target_xy_true[0] > (target_xy_true_x_max / 2):
                # 右转速度
                R_speed = PIDw.cal_output((target_xy_pixel[0] - 320) * K_R_L)
                # R_speed = target_xy_pixel[0] * K_R_L - 1
                # 后退速度
                # B_speed = (target_depth-Math_cm) * K_B
                # B_speed = float(target_depth) - 1
                # B_speed = target_depth - 3
                # B_speed = float(target_depth) - 2
                B_speed = PID.cal_output(target_depth)
                if B_speed < -1:
                    B_speed = -1
                print('右转后退:右转速度:{},后退速度{}'.format(R_speed, B_speed))
                state = unitree_robot.robot_walking(gaitType=1, forwardSpeed=B_speed, sidewaySpeed=0.0, rotateSpeed=-R_speed,
                                                speedLevel=0, bodyHeight=0.0)
            elif target_xy_true[0] < 0:
            #elif target_xy_true[0] < (target_xy_true_x_max / 2):
                # 左转速度
                L_speed = PIDw.cal_output((target_xy_pixel[0] - 320) * K_R_L)
                # L_speed = target_xy_pixel[0] * K_R_L - 1
                # 后退速度
                # B_speed = (target_depth-Math_cm) * K_B
                # B_speed = float(target_depth) - 1
                # B_speed = target_depth - 3
                # B_speed = float(target_depth) - 2
                B_speed = PID.cal_output(target_depth)
                if B_speed < -1:
                    B_speed = -1
                print('左转后退:左转速度:{},后退速度{}'.format(L_speed, B_speed))
                state = unitree_robot.robot_walking(gaitType=1, forwardSpeed=B_speed, sidewaySpeed=0.0, rotateSpeed=-L_speed,
                                                speedLevel=0, bodyHeight=0.0)
            else:
                R_speed = 0
                # F_speed = 0
                L_speed = 0
                # B_speed = 0
                print('静止不动:右转速度:' + R_speed +'左转速度:' + L_speed + '前进速度:' + F_speed + '后退速度:' + B_speed)
                state = unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0, rotateSpeed=0.0,
                                                    speedLevel=0, bodyHeight=0.0)
    return R_speed,F_speed,L_speed,B_speed