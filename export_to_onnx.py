import torch
from pathlib import Path
from args import args_train
from train import JointPrediction
from datasets.joint_graph_dataset import JointGraphDataset
import onnx

def export_to_onnx(args, model):
    num_nodes1, num_edges1, num_nodes2, num_edges2 = 5, 5, 10, 10
    dummy_g1_grid = torch.randn(num_nodes1, JointGraphDataset.grid_size, JointGraphDataset.grid_size, JointGraphDataset.grid_channels)
    dummy_g1_ent = torch.randn(num_nodes1, JointGraphDataset.ent_feature_size)
    dummy_g1_ent[:, 1] = torch.randint(0, len(JointGraphDataset.curve_type_map), [num_nodes1]) 
    dummy_g1_edge_index = torch.randint(0, num_nodes1, (2, num_edges1))
    dummy_g2_grid = torch.randn(num_nodes2, JointGraphDataset.grid_size, JointGraphDataset.grid_size, JointGraphDataset.grid_channels)
    dummy_g2_ent = torch.randn(num_nodes2, JointGraphDataset.ent_feature_size)
    dummy_g2_ent[:, 1] = torch.randint(0, len(JointGraphDataset.curve_type_map), [num_nodes2]) 
    dummy_g2_edge_index = torch.randint(0, num_nodes2, (2, num_edges2))
    dummy_jg_edge_index = torch.randint(0, max(num_nodes1, num_nodes2), (2, num_nodes1 * num_nodes2))
    dummy_num_nodes = torch.tensor([[num_nodes1], [num_nodes2]])
    input = (
        dummy_g1_grid,
        dummy_g1_ent,
        dummy_g1_edge_index,
        dummy_g2_grid,
        dummy_g2_ent,
        dummy_g2_edge_index,
        dummy_jg_edge_index,
        dummy_num_nodes
    )

    input_names = [
        'g1_grid', 
        'g1_ent', 
        'g1_edge_index', 
        'g2_grid', 
        'g2_ent', 
        'g2_edge_index', 
        'jg_edge_index',
        'num_nodes'
    ]

    dynamic_axes = {
        'g1_grid': {0: 'num_nodes1'},
        'g1_ent': {0: 'num_nodes1'},
        'g1_edge_index': {1: 'num_edges1'},
        'g2_grid': {0: 'num_nodes2'},
        'g2_ent': {0: 'num_nodes2'},
        'g2_edge_index': {1: 'num_edges2'},
        'jg_edge_index': {1: 'num_nodes1 * num_nodes2'},
        'output': {0: 'num_nodes1 * num_nodes2'}
    }

    exp_dir = Path(args.exp_dir)
    exp_name_dir = exp_dir / args.exp_name
    file = exp_name_dir / f"{args.checkpoint}.onnx"
    torch.onnx.export(
        model, 
        input, 
        file,
        export_params=True, 
        opset_version=16,  
        do_constant_folding=True,
        input_names=input_names, 
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    onnx_model = onnx.load(file)
    onnx.checker.check_model(onnx_model, full_check=True)
    print("The model is successfully exported and is valid.")

if __name__ == "__main__":
    args = args_train.get_args()
    
    exp_dir = Path(args.exp_dir)
    exp_name_dir = exp_dir / args.exp_name
    checkpoint_file = exp_name_dir / f"{args.checkpoint}.ckpt"
    model = JointPrediction.load_from_checkpoint(checkpoint_file).model
    model.eval()
    model.cpu()
    export_to_onnx(args, model)