### train
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_00 --max_nodes_per_batch 8192 --num_workers 16
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_01 --max_nodes_per_batch 8192 --num_workers 16 --batch_norm False
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_02 --max_nodes_per_batch 8192 --num_workers 16 --loss bce
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_03 --max_nodes_per_batch 8192 --num_workers 16 --loss symmetric
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_04 --max_nodes_per_batch 8192 --num_workers 16 --loss focal
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_05 --max_nodes_per_batch 8192 --num_workers 16 --mpn_layer_num 4
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_06 --max_nodes_per_batch 8192 --num_workers 16 --train_label_scheme Joint,JointEquivalent
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_07 --max_nodes_per_batch 8192 --num_workers 16 --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_08 --max_nodes_per_batch 8192 --num_workers 16 --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --random_rotate
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_09 --max_nodes_per_batch 8192 --num_workers 16 --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --feature_embedding
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_10 --max_nodes_per_batch 8192 --num_workers 16 --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --loss symmetric --train_label_scheme Joint,JointEquivalent
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_11 --max_nodes_per_batch 8192 --num_workers 16 --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --label_smoothing 0.1
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_12 --max_nodes_per_batch 8192 --num_workers 16 --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --feature_embedding --label_smoothing 0.1
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_13 --max_nodes_per_batch 8192 --num_workers 16  --delete_cache --skip_far
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_14 --max_nodes_per_batch 8192 --num_workers 16  --delete_cache --skip_interference
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_15 --max_nodes_per_batch 8192 --num_workers 16  --delete_cache --skip_nurbs
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_16 --max_nodes_per_batch 8192 --num_workers 16  --delete_cache --without_synthetic
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_17 --max_nodes_per_batch 8192 --num_workers 16  --delete_cache --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --joint_type Coincident
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_18 --max_nodes_per_batch 8192 --num_workers 16  --delete_cache --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --joint_type Concentric
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_19 --max_nodes_per_batch 8192 --num_workers 16 --input_features axis_dir,entity_types,area,circumference,param_1,param_2,length,radius,start_point,middle_point,end_point
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_20 --max_nodes_per_batch 8192 --num_workers 16 --input_features axis_dir,entity_types,area,circumference,param_1,param_2,length,radius,start_point,middle_point,end_point --label_smoothing 0.1
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_21 --max_nodes_per_batch 8192 --num_workers 16 --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --feature_embedding --train_label_scheme Joint,JointEquivalent
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_22 --max_nodes_per_batch 8192 --num_workers 16  --delete_cache --skip_interference --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --label_smoothing 0.1
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_23 --max_nodes_per_batch 8192 --num_workers 16 --input_features points,normals,tangents,trimming_mask --label_smoothing 0.1
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_24 --max_nodes_per_batch 8192 --num_workers 16 --input_features points,normals,tangents,trimming_mask --label_smoothing 0.1 --random_rotate
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_25 --max_nodes_per_batch 8192 --num_workers 16 --delete_cache --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --type_head
python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_26 --max_nodes_per_batch 8192 --num_workers 16 --delete_cache --input_features entity_types,area,length,points,normals,tangents,trimming_mask --label_smoothing 0.1

### test (random)
# python train.py --traintest randomtest --dataset data/zw3d-joinable-dataset  --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (pretrained)
# python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (zw3d_test)
# python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_00