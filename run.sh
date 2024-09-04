### train
python train.py --exp_name 00_baseline
python train.py --exp_name 01_no_gat --n_layer_gat 0
python train.py --exp_name 02_no_sat --n_layer_sat 0
python train.py --exp_name 03_no_cat --n_layer_cat 0
python train.py --exp_name 04_no_head1 --n_layer_head1 0
python train.py --exp_name 05_with_type --with_type
python train.py --exp_name 06_less_head --n_head 4
python train.py --exp_name 07_less_embd --n_embd 192
python train.py --exp_name 08_dropout --dropout 0.2
python train.py --exp_name 19_with_bias --bias
python train.py --exp_name 10_more_lr --lr 0.001
python train.py --exp_name 11_equivalent --train_label_scheme Joint,JointEquivalent
python train.py --exp_name 12_bce --loss bce
python train.py --exp_name 13_loss_sym --loss_sym
python train.py --exp_name 14_label_smoothing --label_smoothing 0.1
python train.py --exp_name 15_rotate  --random_rotate

### test (random)
# python train.py --traintest randomtest --dataset data/zw3d-joinable-dataset  --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (pretrained)
# python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (zw3d_test)
# python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_00