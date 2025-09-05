from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
# sys.path.append("/home/robot112/FairMOT-master/src/unitree_legged_sdk/scripts")
# import Unitree_Python_sdk
sys.path.append("/home/robot112/fastermot/src/unitree_legged_sdk-go1/example_py")
import Unitree_Robot as Unitree_Python_sdk
import torch
import numpy as np
#/home/robot112/FairMOT-master/src/unitree_legged_sdk/scripts
import math
# src.unitree_legged_sdk.scripts import Unitree_Python_sdk

## 四足函数调用
unitree_robot = Unitree_Python_sdk.Unitree_Robot()

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
                # 右转速度
                R_speed = target_xy_pixel[0] * K_R_L - 1
                #前进速度
                # F_speed = (target_depth-Math_cm) * K_F
                # F_speed = float(target_depth) - 1
                #F_speed = target_depth - 3
                # F_speed = float(target_depth - 2) / 3
                F_speed = float(((target_depth-2)**2)/9)
                if F_speed > 1:
                    F_speed = 1
                print('右转前进:右转速度:{},前进速度{}'.format(R_speed,F_speed))
                state = unitree_robot.robot_walking(gaitType=1, forwardSpeed=F_speed, sidewaySpeed=0.0, rotateSpeed=-R_speed,
                                                speedLevel=0, bodyHeight=0.0)
            elif target_xy_true[0] < 0:
            #elif target_xy_true[0] < (target_xy_true_x_max / 2):
                # 左转速度
                L_speed = target_xy_pixel[0] * K_R_L - 1
                # 前进速度
                # F_speed = (target_depth-Math_cm) * K_F
                # F_speed = float(target_depth) - 1
                #F_speed = target_depth - 3
                # F_speed = float(target_depth - 2) / 3
                F_speed = float(((target_depth - 2) ** 2) / 9)
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
                R_speed = target_xy_pixel[0] * K_R_L - 1
                # 后退速度
                # B_speed = (target_depth-Math_cm) * K_B
                # B_speed = float(target_depth) - 1
                # B_speed = target_depth - 3
                # B_speed = float(target_depth) - 2
                B_speed = float(-((target_depth - 2) ** 2))
                if B_speed < -1:
                    B_speed = -1
                print('右转后退:右转速度:{},后退速度{}'.format(R_speed, B_speed))
                state = unitree_robot.robot_walking(gaitType=1, forwardSpeed=B_speed, sidewaySpeed=0.0, rotateSpeed=-R_speed,
                                                speedLevel=0, bodyHeight=0.0)
            elif target_xy_true[0] < 0:
            #elif target_xy_true[0] < (target_xy_true_x_max / 2):
                # 左转速度
                L_speed = target_xy_pixel[0] * K_R_L - 1
                # 后退速度
                # B_speed = (target_depth-Math_cm) * K_B
                # B_speed = float(target_depth) - 1
                # B_speed = target_depth - 3
                # B_speed = float(target_depth) - 2
                B_speed = float(-((target_depth - 2) ** 2))
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1,1).expand(N,M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1,-1).expand(N,M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def generate_anchors(nGh, nGw, anchor_wh):
    nA = len(anchor_wh)
    yy, xx = np.meshgrid(np.arange(nGh), np.arange(nGw), indexing='ij')

    mesh = np.stack([xx, yy], axis=0)  # Shape 2, nGh, nGw
    mesh = np.tile(np.expand_dims(mesh, axis=0), (nA, 1, 1, 1)) # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = np.tile(np.expand_dims(np.expand_dims(anchor_wh, -1), -1), (1, 1, nGh, nGw))  # Shape nA x 2 x nGh x nGw
    anchor_mesh = np.concatenate((mesh, anchor_offset_mesh), axis=1)  # Shape nA x 4 x nGh x nGw
    return anchor_mesh


def encode_delta(gt_box_list, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                     gt_box_list[:, 2], gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = np.log(gw/pw)
    dh = np.log(gh/ph)
    return np.stack((dx, dy, dw, dh), axis=1)
