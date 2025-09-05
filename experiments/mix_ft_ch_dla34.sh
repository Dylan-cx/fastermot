cd src
python train.py mot --exp_id mix_ft_ch_dla34_fasterdfc_1242_nopre_200epoch --num_epochs 200 --arch 'fasterdfc_1242' --attention_type 'dfc' --load_model '../exp/mot/crowdhuman_fasterdfc1x1_1242_p_nopre/model_last.pth' --data_cfg '../src/lib/cfg/data.json'
cd ..



