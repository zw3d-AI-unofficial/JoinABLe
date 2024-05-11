### train
# python train.py --dataset data/zw3d-joinable-dataset --exp_dir pretrained --exp_name paper --resume --batch_size 8 --num_workers 4 --epochs 200
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_0 --batch_size 8 --num_workers 4 --epochs 50
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_1 --batch_size 8 --num_workers 4 --epochs 50 --train_label_scheme "Joint,JointEquivalent"
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_2 --batch_size 8 --num_workers 4 --epochs 50 --loss bce
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_3 --batch_size 8 --num_workers 4 --epochs 50 --loss symmetric
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_4 --batch_size 8 --num_workers 4 --epochs 50 --loss focal
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_5 --batch_size 8 --num_workers 4 --epochs 50 --delete_cache --input_features "entity_types,length,face_reversed,edge_reversed,area"
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_6 --batch_size 8 --num_workers 4 --epochs 50 --skip_far --delete_cache
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_7 --batch_size 8 --num_workers 4 --epochs 50 --skip_interference --delete_cache
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_8 --batch_size 8 --num_workers 4 --epochs 50 --mpn_layer_num 4
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_9 --batch_size 8 --num_workers 4 --epochs 50 --skip_nurbs --delete_cache
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_10 --batch_size 8 --num_workers 4 --epochs 50 --joint_type "Coincident" --delete_cache
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_11 --batch_size 8 --num_workers 4 --epochs 50


### test (random)
# python train.py --traintest randomtest --dataset data/zw3d-joinable-dataset  --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (pretrained)
# python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (zw3d_test)
python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_10 --joint_type "Coincident" --delete_cache
