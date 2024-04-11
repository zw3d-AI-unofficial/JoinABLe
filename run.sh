# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_0 --batch_size 8 --num_workers 4 --epochs 50
# python train.py --traintest test --dataset data/tester --exp_name zw3d_test_0 --checkpoint last
python train.py --traintest test --dataset data/tester --exp_dir pretrained --exp_name paper --checkpoint last_run_0