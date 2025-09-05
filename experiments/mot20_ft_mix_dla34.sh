cd src
python train.py mot --exp_id mot20_ft_mix_dla34_60epoch --arch 'fasterdfc_1242' --attention_type 'dfc' --load_model '../exp/mot/mix_ft_ch_dla34_fasterdfc_1242_nopre_120epoch/model_80.pth' --num_epochs 60 --lr_step '15' --data_cfg '../src/lib/cfg/mot20.json'
cd ..