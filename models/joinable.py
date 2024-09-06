import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv,global_mean_pool
from torchvision.ops.focal_loss import sigmoid_focal_loss

from utils import metrics
from datasets.joint_graph_dataset import JointGraphDataset


def cnn2d(inp_channels, hidden_channels, out_dim, num_layers=1):
    assert num_layers >= 1
    modules = []
    for i in range(num_layers - 1):
        modules.append(nn.Conv2d(inp_channels if i == 0 else hidden_channels, hidden_channels, kernel_size=3, padding=1))
        modules.append(nn.ELU())
    modules.append(nn.AdaptiveAvgPool2d(1))
    modules.append(nn.Flatten())
    modules.append(nn.Linear(hidden_channels, out_dim, bias=True))
    return nn.Sequential(*modules)


def cnn1d(inp_channels, hidden_channels, out_dim, num_layers=1):
    assert num_layers >= 1
    modules = []
    for i in range(num_layers - 1):
        modules.append(nn.Conv1d(inp_channels if i == 0 else hidden_channels, hidden_channels, kernel_size=3, padding=1))
        modules.append(nn.ELU())
    modules.append(nn.AdaptiveAvgPool1d(1))
    modules.append(nn.Flatten())
    modules.append(nn.Linear(hidden_channels, out_dim, bias=True))
    return nn.Sequential(*modules)


class MLP(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.c_fc    = nn.Linear(args.n_embd, 4 * args.n_embd, bias=args.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * args.n_embd, args.n_embd, bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, n_dim, bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class FaceEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Turn the comma separated string e.g. "points,entity_types,is_face,length"
        # into lists for each feature type, i.e.
        # - Features used in the UV grid, i.e. points, normals, trimming_mask
        # - Features related to each B-Rep entity e.sg. area, length
        feat_lists = JointGraphDataset.parse_input_features_arg(args.input_features, input_feature_type="face")
        _, self.grid_input_features, self.ent_input_features = feat_lists #points，normals，trimming_mask；area，entity_types
        # Calculate the total size of each feature list
        self.grid_feat_size = JointGraphDataset.get_input_feature_size(self.grid_input_features, input_feature_type="face") #700
        self.ent_feat_size = JointGraphDataset.get_input_feature_size(self.ent_input_features, input_feature_type="face")   #7

        # Setup the layers
        n_vocab = int(2**args.n_bits) #2**8 = 256
        n_embd = args.n_embd          #384
        self.n_embd = n_embd
        if self.grid_feat_size > 0:
            channels = self.grid_feat_size // JointGraphDataset.grid_len    # 700//100 = 7
            self.grid_embd = cnn2d(channels, n_embd, n_embd, num_layers=3)  #cnn2d(7, 384, 384, num_layers=3)
        if self.ent_feat_size > 0:   #7
            self.quantize = args.quantize
            if args.quantize:
                self.ent_embd = nn.ModuleDict({
                    'entity_types': nn.Embedding(len(JointGraphDataset.surface_type_map), n_embd),
                    'axis_pos': nn.Embedding(n_vocab, n_embd // 3),
                    'axis_dir': nn.Embedding(n_vocab, n_embd // 3),
                    'bounding_box': nn.Embedding(n_vocab, n_embd // 6),
                    'area': nn.Embedding(n_vocab, n_embd),
                    'circumference': nn.Embedding(n_vocab, n_embd),
                    'param_1': nn.Embedding(n_vocab, n_embd),
                    'param_2': nn.Embedding(n_vocab, n_embd)
                })
            else:
                self.ent_embd = nn.Linear(self.ent_feat_size, n_embd)      #Linear(7, 384)
        self.ln = LayerNorm(n_embd, bias=args.bias)                        #LayerNorm(384, False)
        self.mlp = MLP(args)

    def get_entity_features(self, g, indices):
        """根据self.ent_input_features,对于其他的'area'等,g[input_feature][indices]；对于'entity_types'，还要改成独热编码；拼接
        Get the entity features that were requested"""
        ent_list = []
        for input_feature in self.ent_input_features:          # ['area', 'entity_types']
            feat = g[input_feature][indices]                   #
            # Convert entity types to a one hot encoding
            if input_feature == "entity_types":
                feat = F.one_hot(feat, num_classes=len(JointGraphDataset.surface_type_map))
            # Make all features 2D
            if len(feat.shape) == 1:
                feat = feat.unsqueeze(1)
            ent_list.append(feat)
        return torch.cat(ent_list, dim=1).float()

    def get_grid_features(self, g):
        """根据self.grid_input_features,对于points,normals,trimming_mask,分别取g.x的:3 ; 3:6 ; 6:
        Get the grid features that were requested"""
        grid_list = []
        if "points" in self.grid_input_features:
            grid_list.append(g.x[:, :, :, :3])
        if "normals" in self.grid_input_features:
            grid_list.append(g.x[:, :, :, 3:6])
        if "trimming_mask" in self.grid_input_features:
            grid_list.append(g.x[:, :, :, 6:])

        grid = torch.cat(grid_list, dim=-1)
        grid = grid.permute(0, 3, 1, 2)
        return grid

    def forward_one_graph(self, g):           #处理单个图，并生成节点级别的特征表示
        def _get_face_node_indices(g):
            """Get the indices of graph nodes corresponding to B-rep faces"""
            face_indices = torch.where(g["is_face"] > 0.5)[0].long()   #找出g["is_face"] 中所有大于 0.5 的元素的索引，并返回这些索引作为结果
            # g["is_face"] tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
            return face_indices               #通过检查图的节点属性 "is_face" 来确定哪些节点对应于 B-rep 面

        face_node_indices = _get_face_node_indices(g)  #获取 B-rep 面对应的节点索引 
        device = g.edge_index.device               
        x = torch.zeros(g.num_nodes, self.n_embd, dtype=torch.float, device=device) #(36,384)

        if self.grid_feat_size > 0:                    # 700
            grid = self.get_grid_features(g)           # torch.Size([36, 7, 10, 10]) ;points,normals,trimming_mask都有的话，就是g.x
            grid_faces = grid[face_node_indices, :]    # 从图 g 中提取网格特征，并仅保留 B-rep 面对应的网格特征
            x[face_node_indices, :] += self.grid_embd(grid_faces) #grid_faces进行cnn2d后，将结果加到 x 张量（0）中对应的位置上
        if self.ent_feat_size > 0:                     # 7
            if self.quantize:                          # quantize为True
                for key in self.ent_input_features:
                    if key in self.ent_embd:           # 如果存在对应的嵌入层 self.ent_embd[key]，则使用这个嵌入层处理实体特征
                        embedding = self.ent_embd[key](g[key][face_node_indices])
                        x[face_node_indices, :] += embedding.reshape(embedding.shape[0], -1)
            else:
                ent_faces = self.get_entity_features(g, face_node_indices)
                x[face_node_indices, :] += self.ent_embd(ent_faces)
        x = x + self.mlp(self.ln(x))            #层归一、多层感知机mlp处理、与原始张量相加。
        return x

    def forward(self, g1, g2):
        x1 = self.forward_one_graph(g1)
        x2 = self.forward_one_graph(g2)
        return x1, x2


class EdgeEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Turn the comma separated string e.g. "points,entity_types,is_face,length"
        # into lists for each feature type, i.e.
        # - Features used in the UV grid, i.e. points, normals, trimming_mask
        # - Features related to each B-Rep entity e.g. area, length
        feat_lists = JointGraphDataset.parse_input_features_arg(args.input_features, input_feature_type="edge")
        _, self.grid_input_features, self.entity_input_features = feat_lists
        # Calculate the total size of each feature list
        self.grid_feat_size = JointGraphDataset.get_input_feature_size(self.grid_input_features, input_feature_type="edge") #60
        self.ent_feat_size = JointGraphDataset.get_input_feature_size(self.entity_input_features, input_feature_type="edge")#5

        # Setup the layers
        n_vocab = int(2**args.n_bits) #256
        n_embd = args.n_embd          #384
        self.n_embd = n_embd
        if self.grid_feat_size > 0:
            channels = self.grid_feat_size // JointGraphDataset.grid_size     #6
            self.grid_embd = cnn1d(channels, n_embd, n_embd, num_layers=3)    #cnn1d(6, 384, 384, num_layers=3)
        if self.ent_feat_size > 0:
            self.quantize = args.quantize
            if self.quantize:
                self.ent_embd = nn.ModuleDict({
                    'entity_types': nn.Embedding(len(JointGraphDataset.curve_type_map), n_embd),
                    'axis_pos': nn.Embedding(n_vocab, n_embd // 3),
                    'axis_dir': nn.Embedding(n_vocab, n_embd // 3),
                    'bounding_box': nn.Embedding(n_vocab, n_embd // 6),
                    'length': nn.Embedding(n_vocab, n_embd),
                    'radius': nn.Embedding(n_vocab, n_embd),
                    'start_point': nn.Embedding(n_vocab, n_embd // 3),
                    'middle_point': nn.Embedding(n_vocab, n_embd // 3),
                    'end_point': nn.Embedding(n_vocab, n_embd // 3)
                })
            else:
                self.ent_embd = nn.Linear(self.ent_feat_size, n_embd)       #nn.Linear(5, 384) 
        self.ln = LayerNorm(n_embd, bias=args.bias)                         #归一化
        self.mlp = MLP(args)
    
    def get_entity_features(self, g, indices):
        """Get the entity features that were requested"""
        ent_list = []
        for input_feature in self.entity_input_features:
            feat = g[input_feature][indices]
            if input_feature == "entity_types":
                # Convert entity types to a one hot encoding
                feat = F.one_hot(feat, num_classes=len(JointGraphDataset.curve_type_map))
            # Make all features 2D
            if len(feat.shape) == 1:
                feat = feat.unsqueeze(1)
            ent_list.append(feat)
        return torch.cat(ent_list, dim=1).float()

    def get_grid_features(self, g):
        """Get the grid features that were requested"""
        # Edge grid features are repeated along the rows,
        # so we only use the feature in the first row
        grid_list = []
        if "points" in self.grid_input_features:
            grid_list.append(g.x[:, 0, :, :3])
        if "tangents" in self.grid_input_features:
            grid_list.append(g.x[:, 0, :, 3:6])
        grid = torch.cat(grid_list, dim=-1)
        grid = grid.permute(0, 2, 1)
        return grid

    def forward_one_graph(self, g):
        def _get_edge_node_indices(g):
            """Get the indices of graph nodes corresponding to B-rep edges"""
            edge_indices = torch.where(g["is_face"] <= 0.5)[0].long()
            return edge_indices
        edge_node_indices = _get_edge_node_indices(g)
        device = g.edge_index.device
        x = torch.zeros(g.num_nodes, self.n_embd, dtype=torch.float, device=device)

        if self.grid_feat_size > 0:
            grid = self.get_grid_features(g)
            grid_edges = grid[edge_node_indices, :]
            x[edge_node_indices, :] += self.grid_embd(grid_edges)
        if self.ent_feat_size > 0:
            if self.quantize:
                for key in self.entity_input_features:
                    if key in self.ent_embd:
                        embedding = self.ent_embd[key](g[key][edge_node_indices])
                        x[edge_node_indices, :] += embedding.reshape(embedding.shape[0], -1)
            else:
                ent_edges = self.get_entity_features(g, edge_node_indices)
                x[edge_node_indices, :] += self.ent_embd(ent_edges)
        x = x + self.mlp(self.ln(x))
        return x    

    def forward(self, g1, g2):
        x1 = self.forward_one_graph(g1)
        x2 = self.forward_one_graph(g2)
        return x1, x2


class GATBlock(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        n_embd = args.n_embd           #384
        self.ln_1 = LayerNorm(n_embd, bias=args.bias)                   # 第一层规范化层，用于归一化输入特征
        self.attn = GATv2Conv(n_embd, n_embd // args.n_head, heads=args.n_head, dropout=args.dropout) #384,48,8,0.0  384/8个头=48每个特征维度
        self.ln_2 = LayerNorm(n_embd, bias=args.bias)                   # 第二层规范化层
        self.mlp = MLP(args)

    def forward(self, x, edges_idx):                                    # [nodes, features] ，边的索引矩阵[2, num_edges]
        x = x + self.attn(self.ln_1(x), edges_idx)                      # 图注意力卷积 (attn)，残差链接
        x = x + self.mlp(self.ln_2(x))
        return x                                                        # [nodes, features]


class SelfAttention(nn.Module):

    def __init__(self, args):
        super().__init__()
        assert args.n_embd % args.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias) # 将输入的嵌入维度 args.n_embd 映射到三倍的嵌入维度。这三个嵌入分别用于查询（query）、键（key）和值（value）
        # output projection
        self.c_proj = nn.Linear(args.n_embd, args.n_embd, bias=args.bias) #将多头注意力的输出映射回原始的嵌入维度
        # regularization
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.dropout = args.dropout

    def forward(self, x, attn_mask):
        B, T, C = x.size() # batch size批次大小, sequence length序列长度, embedding dimensionality (n_embd)嵌入维度  torch.Size([1, 36, 384])

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) nh*hs = 384 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) torch.Size([1, 8, 36, 48])
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attention: 
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side 多头注意力的结果重新组装起来，使其恢复到原始的形状，即每个头部的输出被拼接到一起形成一个完整的输出

        # output projection
        y = self.resid_dropout(self.c_proj(y))           #对多头注意力的输出进行线性变换，并应用残留丢弃层
        return y


class SATBlock(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.ln_1 = LayerNorm(args.n_embd, bias=args.bias)
        self.attn = SelfAttention(args)
        self.ln_2 = LayerNorm(args.n_embd, bias=args.bias)
        self.mlp = MLP(args)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class CrossAttention(nn.Module):

    def __init__(self, args):
        super().__init__()
        assert args.n_embd % args.n_head == 0                           #确保了嵌入维度 n_embd 能够被头的数量 n_head 整除
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias) #确保了嵌入维度 n_embd 能够被头的数量 n_head 整除
        self.c_attn = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias)
        # output projection
        self.c_proj = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)     #将注意力的结果映射回原始的嵌入维度
        # regularization
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.dropout = args.dropout

    def forward(self, x1, x2, attn_mask):
        B, T1, C = x1.size() # batch size, sequence length, embedding dimensionality (n_embd)
        B, T2, C = x2.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q1, k1, v1  = self.c_attn(x1).split(self.n_embd, dim=2)
        k1 = k1.view(B, T1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T1, hs)
        q1 = q1.view(B, T1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T1, hs)
        v1 = v1.view(B, T1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T1, hs)
        q2, k2, v2  = self.c_attn(x2).split(self.n_embd, dim=2)
        k2 = k2.view(B, T2, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T2, hs)
        q2 = q2.view(B, T2, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T2, hs)
        v2 = v2.view(B, T2, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T2, hs)

        # cross-attention: 
        y1 = torch.nn.functional.scaled_dot_product_attention(q1, k2, v2, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)
        y1 = y1.transpose(1, 2).contiguous().view(B, T1, C) # re-assemble all head outputs side by side 组装成原始形状
        y1 = self.resid_dropout(self.c_proj(y1))

        attn_mask=attn_mask.transpose(2, 3)
        y2 = torch.nn.functional.scaled_dot_product_attention(q2, k1, v1, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)
        y2 = y2.transpose(1, 2).contiguous().view(B, T2, C) # re-assemble all head outputs side by side
        y2 = self.resid_dropout(self.c_proj(y2))
        return y1, y2


class CATBlock(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.ln_1 = LayerNorm(args.n_embd, bias=args.bias)
        self.attn = CrossAttention(args)
        self.ln_2 = LayerNorm(args.n_embd, bias=args.bias)
        self.mlp = MLP(args)

    def forward(self, x1, x2, attn_mask_cross):
        x = self.attn(self.ln_1(x1), self.ln_1(x2), attn_mask_cross)
        x1 = x1 + x[0]
        x2 = x2 + x[1]
        x1 = x1 + self.mlp(self.ln_2(x1))
        x2 = x2 + self.mlp(self.ln_2(x2))
        return x1, x2


class MLPBlock(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.ln = LayerNorm(args.n_embd, bias=args.bias)
        self.mlp = MLP(args)

    def forward(self, x):
        x = x + self.mlp(self.ln(x))
        return x


class JointPredictHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.block_list = nn.ModuleList([MLPBlock(args) for _ in range(args.n_layer_head1)]) #指定了要创建的 MLPBlock 的数量。每个 MLPBlock 都是由 args 参数初始化
        self.ln = LayerNorm(args.n_embd, bias=args.bias)
        self.out = nn.Linear(args.n_embd, 1, bias=args.bias) #输出层的输出维度为 1

    def forward(self, x, jg):                          #x,torch.Size([204, 384])
        src, tgt = jg.edge_index[0].long(), jg.edge_index[1].long() #edge_index [2, num_edges];提取出边的源节点索引和目标节点索引
        x = x[src, :] + x[tgt, :]                      #根据节点索引，从张量x提取对应行
        for block in self.block_list:
            x = block(x)
        logits = self.out(self.ln(x)).squeeze()        #torch.Size([6048])
        return logits


class JointTypeHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.block_list = nn.ModuleList([MLPBlock(args) for _ in range(args.n_layer_head2)])
        self.ln = LayerNorm(args.n_embd, bias=args.bias)
        self.out = nn.Linear(args.n_embd, len(JointGraphDataset.joint_type_map), bias=args.bias)

    def forward(self, x, jg):
        ids = torch.where(jg.edge_attr == 1)[0].long()
        src, tgt = jg.edge_index[0].long(), jg.edge_index[1].long()
        x = x[src[ids], :] + x[tgt[ids], :]
        for block in self.block_list:
            x = block(x)
        logits = self.out(self.ln(x))
        return logits


class JoinABLe(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.face_embd = FaceEmbedding(args)
        self.edge_embd = EdgeEmbedding(args)
        self.drop = nn.Dropout(args.dropout) #0
        self.gat_list = nn.ModuleList([GATBlock(args) for _ in range(args.n_layer_gat)])
        self.sat_list = nn.ModuleList([SATBlock(args) for _ in range(args.n_layer_sat)])
        self.cat_list = nn.ModuleList([CATBlock(args) for _ in range(args.n_layer_cat)])
        self.head1 = JointPredictHead(args)
        self.head2 = JointTypeHead(args)
        self.with_type = args.with_type

        self.apply(self._init_weights)

        # 新增特征融合后的池化
        self.pooling1 = torch.mean
        self.pooling2 = torch.mean
        # Define the classifier
        self.classifier = nn.ModuleList([
            nn.Linear(args.n_embd * 2, args.n_embd),
            nn.ReLU(),
            nn.Linear(args.n_embd, 1),
            nn.Dropout(args.dropout),
            nn.Sigmoid()
        ])


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def split_and_pad(self, x, node_counts):
        x_list = torch.split(x, node_counts)
        max_nodes = max(node_counts)

        padded_x_list = []
        for x in x_list:
            n_nodes = x.size(0)
            padding = (0, 0, 0, max_nodes - n_nodes)
            padded_x = F.pad(x, padding, value=0)
            padded_x_list.append(padded_x)
        return torch.stack(padded_x_list)

    def get_attn_masks(self, x1, x2, n_nodes1, n_nodes2):
        B, T1, _ = x1.shape    #torch.Size([1, 36, 384])
        B, T2, _ = x2.shape    #torch.Size([1, 168, 384])
        attn_mask1 = torch.ones((B, 1, T1, T1), dtype=torch.float32, device=x1.device)
        attn_mask2 = torch.ones((B, 1, T2, T2), dtype=torch.float32, device=x1.device)
        attn_mask_cross = torch.ones((B, 1, T1, T2), dtype=torch.float32, device=x1.device)
        for i in range(B):
            attn_mask1[i, 0, :n_nodes1[i], :n_nodes1[i]] = 0         #将注意力掩码中与实际节点数相对应的部分设置为0
            attn_mask2[i, 0, :n_nodes2[i], :n_nodes2[i]] = 0
            attn_mask_cross[i, 0, :n_nodes1[i], :n_nodes2[i]] = 0
        A_SMALL_NUMBER = -1e9
        attn_mask1 *= A_SMALL_NUMBER
        attn_mask2 *= A_SMALL_NUMBER
        attn_mask_cross *= A_SMALL_NUMBER
        return attn_mask1, attn_mask2, attn_mask_cross

    def unpad_and_concat(self, x1, x2, n_nodes1, n_nodes2):
        concat_x = []
        for i in range(len(n_nodes1)):
            size1_i = n_nodes1[i]
            size2_i = n_nodes2[i]
            # Concatenate features from graph1 and graph2 in a interleaved fashion
            # as this is the format that the joint graph expects
            x1_i = x1[i, :size1_i, :]
            x2_i = x2[i, :size2_i, :]
            concat_x.append(x1_i)
            concat_x.append(x2_i)
        x = torch.cat(concat_x, dim=0)
        return x
    
    def forward(self, g1, g2, jg):                             #[404,10,10,7] [270,10,10,7] [674,1]
        # Compute the features for the is_face nodes, and set the rest to zero
        x1_face, x2_face = self.face_embd(g1, g2)              #[404,384] [270,384]
        # Compute the features for the NOT is_face nodes, and set the rest to zero
        x1_edge, x2_edge = self.edge_embd(g1, g2)              #[404,384] [270,384]
        # Sum the two features to populate the edge and face features at the right locations
        x1 = self.drop(x1_edge + x1_face)
        x2 = self.drop(x2_edge + x2_face)      
        # Message passing
        for block in self.gat_list:         #GATBlock
            x1 = block(x1, g1.edge_index)                      #[404,384] 多个batch的g1
            x2 = block(x2, g2.edge_index)                      #[270,384] 多个Batch的g2
        # Prepare data for attention layers 
        joint_graph_unbatched = jg.to_data_list() #jg 转换为一个包含多个图的列表  Data(x=[278, 1], Data(x=[396, 1] x值为0 每个batch的g1+g2
        n_nodes1 = [item.num_nodes_graph1 for item in joint_graph_unbatched]    #tensor([138], /tensor([266], device='cuda:0') 不同batch的g1
        n_nodes2 = [item.num_nodes_graph2 for item in joint_graph_unbatched]    #tensor([140], /tensor([130], device='cuda:0') 不同batch的g2
        x1 = self.split_and_pad(x1, n_nodes1)     #方法将 x1 和 x2 分割并填充到适合的形状，以便适应联合图的结构 torch.Size([2, 266, 384])
        x2 = self.split_and_pad(x2, n_nodes2)                                                             # torch.Size([2, 140, 384])
        # Attention layers
        attn_mask1, attn_mask2, attn_mask_cross = self.get_attn_masks(x1, x2, n_nodes1, n_nodes2) #生成注意力掩码，这些掩码用于指导注意力层的操作
        for block in self.sat_list:        #SATBlock自注意力块；注意力掩码
            x1 = block(x1, attn_mask1)
            x2 = block(x2, attn_mask2)
        for block in self.cat_list:        #CAT；跨图注意力
            x1, x2 = block(x1, x2, attn_mask_cross)           #torch.Size([1, 36, 384])  torch.Size([1, 168, 384]); torch.Size([8, 386, 384]) torch.Size([8, 278, 384])

        # # Pass to post-net
        # x = self.unpad_and_concat(x1, x2, n_nodes1, n_nodes2) #合并，并应用必要的去填充操作 torch.Size([204, 384])
        # logits = self.head1(x, jg)                            #生成最终的预测 logits torch.Size([6048]) 36*!68
        # if self.with_type:
        #     type_logits = self.head2(x, jg)
        #     return logits, type_logits
        # return logits

        joint_judge_prediction = self.joint_judge_predict(x1, x2, n_nodes1, n_nodes2)       # float torch.Size([batch_size]) 二分类预测
        return joint_judge_prediction
    
    # 修改：进行二分类，先池化
    def joint_judge_predict(self, x1, x2, n_nodes1, n_nodes2):
        batch_size, _, _ = x1.size()
        outputs = []
        for i in range(batch_size):
            # 提取批次中的单一样本
            size1_i = n_nodes1[i]
            size2_i = n_nodes2[i]
            x1_i = x1[i:i+1, :size1_i, :]
            x2_i = x2[i:i+1, :size2_i, :]                     # batch_size=1
            x1_pooled = self.pooling1(x1_i, dim=1, keepdim=True)  # torch.Size([batch_size, nodes, features]) -> torch.Size([batch_size,1, features])
            x2_pooled = self.pooling1(x2_i, dim=1, keepdim=True)  
            x1_pooled = x1_pooled.squeeze(dim=1) # torch.Size([batch_size, 384])
            x2_pooled = x2_pooled.squeeze(dim=1)
            # Concatenate the pooled features from both graphs
            pooled_features = torch.cat((x1_pooled, x2_pooled), dim=1) # torch.Size([batch_size, 768])
            # Pass through the classifier
            for layer in self.classifier:                              # linear relu linear sigmoid
                pooled_features = layer(pooled_features)               # torch.Size([batch_size, 1]) （0-1）的概率值
            # Convert the final output of the classifier to a probability using Sigmoid
            output = pooled_features.squeeze(0)       # float torch.Size([batch_size]) 清除大小为1的维度
            outputs.append(output)   

        # 使用torch.stack()堆叠所有输出，并挤压新生成的维度
        stacked_outputs = torch.stack(outputs)
        joint_judge_prediction = stacked_outputs.squeeze(1)  # 移除大小为1的新维度

        return joint_judge_prediction

    def soft_cross_entropy(self, input, target):
        logprobs = F.log_softmax(input, dim=-1) #对最后一个维度，对数概率分布 torch.Size([6048])
        # loss = -(target * logprobs).sum()
        loss = -torch.dot(target, logprobs)     #一维张量和一维张量进行点积操作
        return loss

    def compute_loss(self, args, x, joint_graph):     #x : torch.Size([6048]) ；joint_graph : torch.Size([204])
        def label_smoothing(one_hot_labels, factor):  #对标签进行平滑处理;factor平滑因子
            num_classes = one_hot_labels.size(-1)
            return one_hot_labels * (1 - factor) + factor / num_classes
    
        joint_graph_unbatched = joint_graph.to_data_list() 
        batch_size = len(joint_graph_unbatched)
        size_of_each_joint_graph = [np.prod(list(item.edge_attr.shape)) for item in joint_graph_unbatched] #边的数量累乘6048
        num_nodes_graph1 = [item.num_nodes_graph1 for item in joint_graph_unbatched] #节点数量列表
        num_nodes_graph2 = [item.num_nodes_graph2 for item in joint_graph_unbatched]

        # Compute loss individually with each joint in the batch
        loss_clf = 0           #分类损失
        loss_sym = 0           #对称损失
        loss_type = 0          #类型损失
        start = 0

        if args.with_type:     #有类型
            x, type_x = x
            loss_type = F.cross_entropy(type_x, joint_graph.joint_type_list) #交叉熵损失

        for i in range(batch_size):
            size_i = size_of_each_joint_graph[i]
            end = start + size_i
            x_i = x[start:end]
            labels_i = joint_graph_unbatched[i].edge_attr                 #torch.Size([6048]) joint_graph的edge_attr就是lable
            # Label smoothing
            if args.label_smoothing != 0.0:                               #启用标签平滑
                labels_i = label_smoothing(labels_i, args.label_smoothing)
            # Classification loss
            if args.loss == "bce":
                num_total = np.prod(list(joint_graph_unbatched[i].edge_attr.shape))
                num_pos = torch.sum(joint_graph_unbatched[i].edge_attr)
                pos_weight = (num_total - num_pos) // num_pos
                loss_clf += self.bce_loss(x_i, labels_i, pos_weight=pos_weight)
            elif args.loss == "mle":
                loss_clf += self.mle_loss(x_i, labels_i)                   #torch.Size([6048])，
            elif args.loss == "focal":
                loss_clf += self.focal_loss(x_i, labels_i)
            # Symmetric loss
            if args.loss_sym:
                loss_sym += self.symmetric_loss(args, x_i, labels_i, num_nodes_graph1[i], num_nodes_graph2[i]) #对称损失
            start = end

        # Total loss
        loss = loss_clf + loss_sym + loss_type
        loss = loss / float(batch_size)
        return loss

    def mle_loss(self, x, labels):
        # Normalize the ground truth matrix into a PDF
        labels = labels / labels.sum()                 #归一化
        labels = labels.view(-1)                       #展平为一维
        # Compare the predicted and ground truth PDFs
        loss = self.soft_cross_entropy(x, labels)
        return loss

    def focal_loss(self, x, ys_matrix, gamma=2.0, alpha=0.25):
        x = x.unsqueeze(1)
        ys_matrix = ys_matrix.flatten().unsqueeze(1).float()
        return sigmoid_focal_loss(x, ys_matrix, reduction="sum", gamma=gamma, alpha=alpha)

    def bce_loss(self, x, ys_matrix, pos_weight=200.0):
        x = x.unsqueeze(1)
        ys_matrix = ys_matrix.flatten().float().unsqueeze(1)
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.tensor([pos_weight]).to(x.device))
        return loss_fn(x, ys_matrix)

    def symmetric_loss(self, args, x, labels, n1, n2):
        x_2d = x.view(n1, n2)
        labels_2d = labels.view(n1, n2)
        ys_2d = torch.nonzero(labels_2d).to(x.device)
        loss1, loss2 = None, None
        if args.loss == "mle":
            loss1 = F.cross_entropy(x_2d[ys_2d[:, 0], :], ys_2d[:, 1], reduction="mean")
            loss2 = F.cross_entropy(x_2d[:, ys_2d[:, 1]].transpose(0, 1), ys_2d[:, 0], reduction="mean")
        else:
            loss_fn1 = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.tensor([n2 - 1]).to(x.device))
            loss1 = loss_fn1(x_2d[ys_2d[:, 0], :].view(-1, 1), labels_2d[ys_2d[:, 0], :].view(-1, 1).float())

            loss_fn2 = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.tensor([n1 - 1]).to(x.device))
            loss2 = loss_fn2(x_2d[:, ys_2d[:, 1]].view(-1, 1), labels_2d[:, ys_2d[:, 1]].view(-1, 1).float())
        loss = 0.5 * (loss1 + loss2)
        return loss

    def accuracy(self, prob, labels, t):
        gt = labels.view(-1)
        pred = (prob >= t).int()
        true_positive = torch.sum(torch.logical_and(gt == 1, pred == 1)).item()
        false_negative = torch.sum(torch.logical_and(gt == 1, pred == 0)).item()
        false_positive = torch.sum(torch.logical_and(gt == 0, pred == 1)).item()
        true_negative = torch.sum(torch.logical_and(gt == 0, pred == 0)).item()
        return [true_positive, false_negative, false_positive, true_negative]

    def hit_at_top_k(self, prob, ys, k):  #模型的输出概率张量
        pred_idx = torch.argsort(-prob)   # prob 张量中的概率值进行降序排序，并返回排序后的索引
        hit = 0
        for i in range(torch.min([k, prob.shape[0]])):
            if pred_idx[i] in ys:
                hit += 1                  #检查每个预测索引 pred_idx[i] 是否在真实的标签 ys 中出现。
        num_joints = ys.shape[0]
        hit_best = torch.min([num_joints, k])
        return [hit, hit_best, num_joints] #实际命中的次数；最好的命中情况；实际的标签数量

    def precision_at_top_k(self, logits, labels, n1, n2, k=None): #logits torch.Size([6048])；labels torch.Size([6048])；36,168， 评估的前 k 个预测
        logits = logits.view(n1, n2)     #logits重塑36*168
        labels = labels.view(n1, n2)
        if k is None:
            k = metrics.get_k_sequence() #获取一个预定义的 k 值序列
        return metrics.hit_at_top_k(logits, labels, k=k) #计算精确度

    def precision_type(self, logits, labels):
        predicted = torch.argmax(logits, dim=1)
        return torch.sum(predicted == labels).item() / len(labels)