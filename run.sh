### train
python train.py --exp_name 00_baseline
python train.py --exp_name 01_quantize --quantize
python train.py --exp_name 02_less_gat --n_layer_gat 1
python train.py --exp_name 03_less_sat --n_layer_sat 1
python train.py --exp_name 04_less_cat --n_layer_cat 1
python train.py --exp_name 05_no_head1 --n_layer_head1 0
python train.py --exp_name 06_with_type --with_type
python train.py --exp_name 07_less_head --n_head 4
python train.py --exp_name 08_less_embd --n_embd 192
python train.py --exp_name 09_dropout --dropout 0.2
python train.py --exp_name 10_with_bias --bias
python train.py --exp_name 11_more_lr --lr 0.001
python train.py --exp_name 12_equivalent --train_label_scheme Joint,JointEquivalent
python train.py --exp_name 13_bce --loss bce
python train.py --exp_name 14_loss_sym --loss_sym
python train.py --exp_name 15_bce_equ --loss bce --train_label_scheme Joint,JointEquivalent
python train.py --exp_name 16_label_smoothing --label_smoothing 0.1
python train.py --exp_name 17_rotate  --random_rotate

# python train.py --exp_name 18  --delete_cache --skip_far
# python train.py --exp_name 19  --delete_cache --skip_interference
# python train.py --exp_name 20  --delete_cache --skip_nurbs
# python train.py --exp_name 21  --delete_cache --skip_synthetic

### test (random)
# python train.py --traintest randomtest --dataset data/zw3d-joinable-dataset  --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (pretrained)
# python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (zw3d_test)
# python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_00