1.训练消融模型：
sh experiments/mix_mot17_half_dla34.sh

2.消融实验测试：
python src/track_half.py mot --load_model ../exp/mot/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True --arch fasterdfc_1242 --attention_type dfc

3.训练mot17：
sh experiments/crowdhuman_dla34.sh
sh experiments/mix_ft_ch_dla34.sh

4.训练mot20：
1）：取消注释jde.py的：
#np.clip(xy[:, 0], 0, width, out=xy[:, 0])
#np.clip(xy[:, 2], 0, width, out=xy[:, 2])
#np.clip(xy[:, 1], 0, height, out=xy[:, 1])
#np.clip(xy[:, 3], 0, height, out=xy[:, 3])
2）：终端执行：
sh experiments/crowdhuman_dla34.sh
sh experiments/mix_ft_ch_dla34.sh
sh experiments/mot20_ft_mix_dla34.sh

5.测试mot17：
python src/track.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.4 --test_mot17 True --arch fasterdfc_1242 --attention_type dfc

6.测试mot20：
python src/track.py mot --test_mot20 True --load_model your_mot20_model.pth --conf_thres 0.3 --test_mot20 True --arch fasterdfc_1242 --attention_type dfc

opt:
第一次reid匹配阈值opt.match_thres=0.4
第二次iou匹配阈值0.5
第三次低分iou匹配阈值0.4
第四次外观分区匹配阈值0.1
高分检测opt.conf_thres=0.4（mot17）   0.3（mot20）
低分检测0.2
外观分区0.4