from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp

import sys

#python src/track摄像头跟踪.py mot --load_model ./models/mot17_120.pth --conf_thres 0.6

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

# from tracker.multitracker import JDETracker
from tracker.byte_patch import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts

import matplotlib.pyplot as plt
#from lib.utils.utils import xyz_true
from lib.utils.twopid_utils import xyz_true
#from lib.utils.pid_utils import xyz_true
#from lib.utils.para_utils import xyz_true


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    #创建保存地址
    if save_dir:
        mkdir_if_missing(save_dir)
    #初始化函数（调用JDETracker类）
    tracker = JDETracker(opt, frame_rate=frame_rate)
    #初始化时间模块
    timer = Timer()
    #结果数组
    results = []
    #用来保存帧 中心点坐标 目标id
    results_centers = []
    centers = []
    #帧的id
    frame_id = 0

    ####################################################################################################################
    #用来定义要跟踪的行人id
    track_id = 1
    ####################################################################################################################

    #for path, img, img0 in dataloader:
    for i, (path, img, img0, aligned_depth_frame, color_intrin_part) in enumerate(dataloader):
    # for i, (path, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        #加载开始时间
        timer.tic()

        if use_cuda:
            #from_numpy数组转换成张量
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        #得到每一帧的实时检测和跟踪结果
        #获取输出轨迹
        online_targets = tracker.update(path, blob, img0)
        online_tlwhs = []
        online_ids = []
        online_centers = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh   #跟踪框的左上角和宽高
            tid = t.track_id  #跟踪框的id
            vertical = tlwh[2] / tlwh[3] > 1.6   #存放宽*高=面积大于1.6的真值表  0001100等等
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical: #如果面积大于最小box面积并且面积小于1.6的
                online_tlwhs.append(tlwh)  #添加该预测宽高进入到online_tlwhs
                online_ids.append(tid)     #添加该预测id进入到online_ids
                #online_scores.append(t.score)
                #中心点坐标
                online_centers.append([tlwh[0] + tlwh[2] / 2, tlwh[1] - tlwh[3] / 2])
        #输出时间
        timer.toc()
        # save results
        #id w 和 h 目标id
        #w h为 【左上角坐标，宽 高】
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #帧 中心点坐标 目标id
        #results_centers[i][0]= 1 第i+1帧
        #results_centers[i][1]    第i+1帧的所有目标物体的中心点数组
        #results_centers[i][1][j] 第i+1帧的第j目标物体的中心点
        #results_centers[i][1][j][0] 第i+1帧的第j目标物体的中心点的x坐标
        #results_centers[i][1][j][1] 第i+1帧的第j目标物体的中心点的y坐标
        #results_centers[i][2]    第i+1帧的所有目标物体的存在id
        #results_centers[i][2][j] 第i+1帧的第j个目标物体的id是xx
        #ps:后期在进行跟踪的时候,可以通过这个数组进行目标中心点的调用
        results_centers.append((frame_id + 1, online_centers, online_ids))
        #centers.append(online_centers)
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if track_id in results_centers[frame_id][2]:
            #print(results_centers[frame_id][2].index(track_id))
            print('ROS send center plot xy!')
            print('forward to xy')
            print(results_centers[frame_id][1][results_centers[frame_id][2].index(track_id)])
            xyz_true(results_centers[frame_id][1][results_centers[frame_id][2].index(track_id)], aligned_depth_frame, color_intrin_part)
        else:
            print('stop to run!')
        #保存轨迹
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        #显示图像
        if show_image:
            cv2.imshow('online_im', online_im)
            cv2.waitKey(1)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    #用来显示输出的中心点位置坐标图
    for frame_id_show in range(frame_id):
        if track_id in results_centers[frame_id_show][2]:
            plt.plot(results_centers[frame_id_show][1][results_centers[frame_id_show][2].index(track_id)][0], results_centers[frame_id_show][1][results_centers[frame_id_show][2].index(track_id)][1], 'ro')
    plt.show()
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []

    output_dir = os.path.join(data_root, '..', 'outputs', exp_name) if save_images or save_videos else None
    # logger.info('start seq: {}'.format(seq))
    # dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
    # dataloader = datasets.LoadVideo(0, opt.img_size)
    dataloader = datasets.LoadVideo(3, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate
    # result_filename = os.path.join(result_root, '{}.txt'.format(seq))
    # meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
    # frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
    nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                          save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
    n_frame += nf
    timer_avgs.append(ta)
    timer_calls.append(tc)

    # eval
    logger.info('Evaluate seq: {}'.format("test"))
    evaluator = Evaluator(data_root, "test", data_type)
    accs.append(evaluator.eval_file(result_filename))
    if save_videos:
        output_video_path = osp.join(output_dir, '{}.mp4'.format("test"))
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
        os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, "test", metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    data_root = os.path.join(opt.data_dir, 'MOT16/train')
    torch.cuda.empty_cache()

    main(opt,
         data_root=data_root,
         exp_name='MOT17_04',
         show_image=True,
         save_images=False,
         save_videos=True)
