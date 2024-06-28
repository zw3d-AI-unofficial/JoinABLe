### train
python train.py --exp_name test_00
python train.py --exp_name test_01_bce_eqv --loss bce --train_label_scheme Joint,JointEquivalent
python train.py --exp_name test_02_smooth --label_smoothing 0.1
python train.py --exp_name test_03_type --with_type

# python train.py --exp_name test_02 --loss bce
# python train.py --exp_name test_03 --loss symmetric
# python train.py --exp_name test_04 --loss focal
# python train.py --exp_name test_06 --train_label_scheme Joint,JointEquivalent
# python train.py --exp_name test_08 --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --random_rotate
# python train.py --exp_name test_10 --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --loss symmetric --train_label_scheme Joint,JointEquivalent
# python train.py --exp_name test_12 --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --feature_embedding --label_smoothing 0.1
# python train.py --exp_name test_13  --delete_cache --skip_far
# python train.py --exp_name test_14  --delete_cache --skip_interference
# python train.py --exp_name test_15  --delete_cache --skip_nurbs
# python train.py --exp_name test_16  --delete_cache --skip_synthetic
# python train.py --exp_name test_17  --delete_cache --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --joint_type Coincident
# python train.py --exp_name test_18  --delete_cache --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --joint_type Concentric
# python train.py --exp_name test_19 --input_features axis_dir,entity_types,area,circumference,param_1,param_2,length,radius,start_point,middle_point,end_point
# python train.py --exp_name test_20 --input_features axis_dir,entity_types,area,circumference,param_1,param_2,length,radius,start_point,middle_point,end_point --label_smoothing 0.1
# python train.py --exp_name test_21 --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --feature_embedding --train_label_scheme Joint,JointEquivalent
# python train.py --exp_name test_22  --delete_cache --skip_interference --input_features axis_pos,axis_dir,entity_types,area,circumference,param_1,param_2,length,radius --label_smoothing 0.1
# python train.py --exp_name test_23 --input_features points,normals,tangents,trimming_mask --label_smoothing 0.1
# python train.py --exp_name test_24 --input_features points,normals,tangents,trimming_mask --label_smoothing 0.1 --random_rotate
# python train.py --exp_name test_26 --input_features entity_types,area,length,points,normals,tangents,trimming_mask --label_smoothing 0.1

### test (random)
# python train.py --traintest randomtest --dataset data/zw3d-joinable-dataset  --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (pretrained)
# python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (zw3d_test)
# python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_00