cd src
#python train.py mot --exp_id crowdhuman_fasterdfc1x1_1242_p_pre --attention_type 'dfc' --gpus '0,1' --batch_size 24 --arch 'fasterdfc_1242' --load_model '../models/fasternet_t0-epoch.281-val_acc1.71.9180.pth' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'
python train.py mot --exp_id crowdhuman_fasterdfc1x1_1242_p_nopre --attention_type 'dfc' --gpus '0,1' --batch_size 20 --arch 'fasterdfc_1242' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'
cd ..