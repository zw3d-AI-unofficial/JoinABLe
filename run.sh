### train
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test --batch_size 8 --num_workers 4 --epochs 100
# python train.py --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_1 --batch_size 8 --num_workers 4 --epochs 50 --input_features "entity_types,length,face_reversed,edge_reversed,area"
# python train.py --dataset data/zw3d-joinable-dataset --exp_dir pretrained --exp_name paper --checkpoint last --batch_size 8 --num_workers 4 --epochs 200

### test (zw3d_test)
python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_name zw3d_test --checkpoint best
# python train.py --traintest test --dataset data/zw3d-joinable-dataset --exp_name zw3d_test_1  --checkpoint last --input_features "entity_types,length,face_reversed,edge_reversed,area" --test_label_scheme "Joint"

### test (random)
# python train.py --traintest randomtest --dataset data/zw3d-joinable-dataset  --exp_dir pretrained --exp_name paper --checkpoint last_run_0

### test (pretrained)
# python train.py --traintest test --dataset data/zw3d-joinable-dataset  --exp_dir pretrained --exp_name paper --checkpoint last_run_0