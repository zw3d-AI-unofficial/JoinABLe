import onnxruntime as ort
import numpy as np
import torch
from datasets.joint_graph_dataset import JointGraphDataset

# 加载 ONNX 模型
model_path = "results_onnx/demo/last.onnx"
session = ort.InferenceSession(model_path)

# 加载测试数据
tensors_dict = torch.load('tensors_dict.pt')

input_data = {
    'g1_grid': tensors_dict['g1_grid'].numpy(), 
    'g1_ent': tensors_dict['g1_ent'].numpy(), 
    'g2_grid': tensors_dict['g2_grid'].numpy(), 
    'g2_ent': tensors_dict['g2_ent'].numpy(), 
    'jg_edge_index': tensors_dict['jg_edge_index'].numpy(),
    'num_nodes': tensors_dict['num_nodes'].numpy()
}

# 进行推理
output = session.run(['output'], input_data)

# 打印输出
print(torch.topk(torch.tensor(np.array(output)), 5))
