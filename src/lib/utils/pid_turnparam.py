from matplotlib import pyplot as plt
import random
from PID_controller import PID_posi_2

PID = PID_posi_2(k=[0.9, 0.0005, 0.05], target=0, upper=1, lower=-1)

def PIDcon(now_v, expc_v):
    sum_err = 0
    v_list = []
    err = expc_v - now_v
    err_last = err  # 上一次的输出,先初始化为一致
    cnt = 0
    while abs(err) > 0.0001:  # 误差还在范围外
        err = expc_v - now_v
        uk = - PID.cal_output(err)
        # uk = Kp * err + Ki * sum_err + Kd * (err - err_last)  # 确定本次输出
        now_v = now_v + uk + random.uniform(-0.01, 0.01)  # 更新当前速度,加入一个扰动
        # now_v = now_v + uk
        v_list.append(now_v)
        cnt += 1
    print(cnt)
    plt.plot(v_list)
    plt.show()


if __name__ == '__main__':
    # PIDcon(0, 0.7)
    K_R_L = 1 / 320
    target_xy_pixel = 320
    R_speed = target_xy_pixel * K_R_L - 1
    F_speed = 0.33
    if F_speed > 1:
        F_speed = 1
    print('右转前进:右转速度:{},前进速度{}'.format(R_speed, F_speed))

    R_speed1 = - PID.cal_output((target_xy_pixel - 320) * K_R_L)
    print('右转前进:右转速度:{},前进速度{}'.format(R_speed1, F_speed))
