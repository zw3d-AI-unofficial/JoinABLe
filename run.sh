### train
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_00 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_01 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --batch_norm
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_02 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --loss bce
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_03 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --loss symmetric
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_04 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --loss focal
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_05 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --mpn_layer_num 4
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_06 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --input_features entity_types,length,face_reversed,edge_reversed,area
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_07 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --train_label_scheme Joint,JointEquivalent
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_08 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --skip_far --delete_cache
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_09 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --skip_interference --delete_cache
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_10 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --skip_nurbs --delete_cache
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_11 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --without_synthetic --delete_cache
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_12 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --joint_type Coincident --batch_norm --delete_cache
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_13 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --joint_type Concentric --batch_norm --delete_cache
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_14 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --input_features axis_pos,axis_dir,bounding_box,entity_types,area,circumference,param_1,param_2,reversed,length,radius --batch_norm --delete_cache
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_15 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --input_features axis_pos,axis_dir,bounding_box,entity_types,area,circumference,param_1,param_2,reversed,length,radius --batch_norm --random_rotate
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_16 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --input_features axis_pos,axis_dir,bounding_box,entity_types,area,circumference,param_1,param_2,reversed,length,radius --batch_norm --feature_embedding
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_17 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --input_features axis_pos,axis_dir,bounding_box,entity_types,area,circumference,param_1,param_2,reversed,length,radius --batch_norm --loss symmetric --train_label_scheme Joint,JointEquivalent
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_18 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --input_features axis_pos,axis_dir,bounding_box,entity_types,area,circumference,param_1,param_2,reversed,length,radius --batch_norm --label_smoothing 0.1
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_19 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --input_features axis_pos,axis_dir,bounding_box,entity_types,area,circumference,param_1,param_2,reversed,length,radius --batch_norm --feature_embedding --label_smoothing 0.1
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_20 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --input_features axis_pos,axis_dir,bounding_box,entity_types,area,circumference,param_1,param_2,reversed,length,radius --batch_norm --label_smoothing 0.1 --joint_type Concentric --delete_cache
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_21 --max_nodes_per_batch 8192 --num_workers 16 --epochs 50 --input_features axis_pos,axis_dir,bounding_box,entity_types,area,circumference,param_1,param_2,reversed,length,radius --batch_norm --label_smoothing 0.1 --joint_type Coincident --delete_cache

### test (random)
# python train.py --traintest randomtest --dataset data/zw3d-joinable-dataset  --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (pretrained)
# python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (zw3d_test)
# python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_00