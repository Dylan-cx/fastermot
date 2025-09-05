cd src
python train.py mot --exp_id mix_mot17_half_fasterdfc_1242 --arch 'fasterdfc_1242' --attention_type 'dfc' --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/data_half.json'
cd ..