"""
5_gnn_models.py
GNN模型定义

包含:
- GCN (Graph Convolutional Network)
- GraphSAGE
- GAT (Graph Attention Network)
- RGCN (Relational GCN) - 用于异构图
- HGT (Heterogeneous Graph Transformer) - 可选
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, RGCNConv, HGTConv, Linear
from torch_geometric.nn import global_mean_pool


# ==================== Base Model ====================
class BaseGNN(nn.Module):
    """基础GNN框架"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


# ==================== GCN ====================
class GCN(BaseGNN):
    """Graph Convolutional Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# ==================== GraphSAGE ====================
class GraphSAGE(BaseGNN):
    """GraphSAGE with mean aggregation"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# ==================== GAT ====================
class GAT(BaseGNN):
    """Graph Attention Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, dropout=0.5, heads=4):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Output layer (single head)
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# ==================== RGCN (异构图) ====================
class RGCN(nn.Module):
    """Relational GCN for heterogeneous graphs"""
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_relations, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))
    
    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_type)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_type)
        return x


# ==================== 异构图转换器 (用于同构GNN) ====================
class HeteroToHomoWrapper:
    """
    将异构图转换为同构图，以便使用标准GNN模型
    
    策略: 
    1. 将所有节点类型映射到统一的feature space
    2. 合并所有边类型
    """
    @staticmethod
    def convert(hetero_data, target_node_type='drug', metapath=None):
        """
        Args:
            hetero_data: HeteroData
            target_node_type: 目标节点类型 (要预测的节点)
            metapath: 元路径，如 [('drug', 'prescribed_in', 'visit'), 
                                  ('visit', 'diagnosed_with', 'disease')]
        
        Returns:
            x: 节点特征
            edge_index: 边索引
            node_mapping: 节点映射关系
        """
        if metapath is None:
            # 默认: drug -> visit -> disease -> visit -> drug
            metapath = [
                ('drug', 'prescribed_in', 'visit'),
                ('visit', 'diagnosed_with', 'disease'),
                ('disease', 'diagnosed_in', 'visit'),
                ('visit', 'prescribed', 'drug')
            ]
        
        # 简化版本: 只使用drug节点和drug-drug边（通过visit/disease连接）
        # 这里我们构建一个drug的同构图
        
        num_drugs = hetero_data['drug'].num_nodes
        x = hetero_data['drug'].x
        
        # 构建drug-drug边: 通过共享visit/disease来连接
        visit_drug = hetero_data['visit', 'prescribed', 'drug'].edge_index
        visit_disease = hetero_data['visit', 'diagnosed_with', 'disease'].edge_index
        
        # drug -> visit mapping
        drug_to_visits = {}
        for i in range(visit_drug.shape[1]):
            visit_idx = visit_drug[0, i].item()
            drug_idx = visit_drug[1, i].item()
            if drug_idx not in drug_to_visits:
                drug_to_visits[drug_idx] = set()
            drug_to_visits[drug_idx].add(visit_idx)
        
        # visit -> disease mapping
        visit_to_diseases = {}
        for i in range(visit_disease.shape[1]):
            visit_idx = visit_disease[0, i].item()
            disease_idx = visit_disease[1, i].item()
            if visit_idx not in visit_to_diseases:
                visit_to_diseases[visit_idx] = set()
            visit_to_diseases[visit_idx].add(disease_idx)
        
        # 构建drug-drug边
        edges = []
        for drug1 in range(num_drugs):
            if drug1 not in drug_to_visits:
                continue
            visits1 = drug_to_visits[drug1]
            
            # 通过共享的disease连接
            diseases1 = set()
            for v in visits1:
                if v in visit_to_diseases:
                    diseases1.update(visit_to_diseases[v])
            
            for drug2 in range(drug1 + 1, num_drugs):
                if drug2 not in drug_to_visits:
                    continue
                visits2 = drug_to_visits[drug2]
                
                diseases2 = set()
                for v in visits2:
                    if v in visit_to_diseases:
                        diseases2.update(visit_to_diseases[v])
                
                # 如果共享disease，则连接
                if len(diseases1 & diseases2) > 0:
                    edges.append([drug1, drug2])
                    edges.append([drug2, drug1])  # 无向图
        
        if len(edges) == 0:
            # 如果没有边，创建一个自环图
            edge_index = torch.tensor([[i, i] for i in range(num_drugs)], dtype=torch.long).t()
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        return x, edge_index


# ==================== 模型工厂 ====================
def create_model(model_name, in_channels, hidden_channels, out_channels, 
                 num_layers=2, dropout=0.5, **kwargs):
    """
    模型工厂函数
    
    Args:
        model_name: 'GCN', 'GraphSAGE', 'GAT', 'RGCN'
        in_channels: 输入特征维度
        hidden_channels: 隐藏层维度
        out_channels: 输出维度（类别数）
        num_layers: 层数
        dropout: dropout率
    """
    model_name = model_name.upper()
    
    if model_name == 'GCN':
        return GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
    
    elif model_name == 'GRAPHSAGE' or model_name == 'SAGE':
        return GraphSAGE(in_channels, hidden_channels, out_channels, num_layers, dropout)
    
    elif model_name == 'GAT':
        heads = kwargs.get('heads', 4)
        return GAT(in_channels, hidden_channels, out_channels, num_layers, dropout, heads)
    
    elif model_name == 'RGCN':
        num_relations = kwargs.get('num_relations', 4)
        return RGCN(in_channels, hidden_channels, out_channels, num_relations, num_layers, dropout)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==================== 测试 ====================
if __name__ == "__main__":
    # Test models
    in_channels = 4
    hidden_channels = 64
    out_channels = 4
    num_nodes = 100
    num_edges = 500
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    print("Testing GNN models...")
    
    # GCN
    model = create_model('GCN', in_channels, hidden_channels, out_channels)
    out = model(x, edge_index)
    print(f"✓ GCN output shape: {out.shape}")
    
    # GraphSAGE
    model = create_model('GraphSAGE', in_channels, hidden_channels, out_channels)
    out = model(x, edge_index)
    print(f"✓ GraphSAGE output shape: {out.shape}")
    
    # GAT
    model = create_model('GAT', in_channels, hidden_channels, out_channels, heads=4)
    out = model(x, edge_index)
    print(f"✓ GAT output shape: {out.shape}")
    
    print("\n✓ All models tested successfully!")
