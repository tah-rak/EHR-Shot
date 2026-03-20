"""
4_data_preparation.py
将Project 1的异构图转换为GNN训练数据

预测任务: Drug Classification (4-class)
- Quadrant 1: Chronic, Broad-Spectrum
- Quadrant 2: Chronic, Specialized  
- Quadrant 3: Acute, Broad-Spectrum
- Quadrant 4: Acute, Specialized
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

class GNNDataPreparator:
    def __init__(self, graph_builder):
        """
        Args:
            graph_builder: 来自Project 1的HeterogeneousGraphBuilder实例
        """
        self.gb = graph_builder
        self.hetero_data = graph_builder.hetero_data
        
    def create_drug_labels(self):
        """
        根据Broad Spectrum (BS) 和 Treatment Persistence (TP) 创建药物分类标签
        
        Returns:
            labels: torch.Tensor, shape [num_drugs]
            label_names: dict, 标签名称映射
        """
        print("\n[Data Prep] Creating drug classification labels...")
        
        # 计算每个drug的BS和TP
        drug_stats = []
        
        # 从边中提取drug的统计信息
        visit_drug_edges = self.hetero_data['visit', 'prescribed', 'drug'].edge_index
        visit_disease_edges = self.hetero_data['visit', 'diagnosed_with', 'disease'].edge_index
        visit_patient_edges = self.hetero_data['visit', 'belongs_to', 'patient'].edge_index
        
        for drug_idx in range(self.hetero_data['drug'].num_nodes):
            # 找到与该drug相关的所有visit
            drug_mask = visit_drug_edges[1] == drug_idx
            visits_for_drug = visit_drug_edges[0][drug_mask].unique()
            
            # Broad Spectrum: 通过这些visits能到达多少个unique diseases
            diseases_set = set()
            for visit in visits_for_drug:
                diseases_at_visit = visit_disease_edges[1][visit_disease_edges[0] == visit]
                diseases_set.update(diseases_at_visit.tolist())
            bs_score = len(diseases_set)
            
            # Treatment Persistence: visits数量 / unique patients数量
            patients_set = set()
            for visit in visits_for_drug:
                patients_at_visit = visit_patient_edges[1][visit_patient_edges[0] == visit]
                patients_set.update(patients_at_visit.tolist())
            tp_score = len(visits_for_drug) / max(len(patients_set), 1)
            
            drug_stats.append({
                'drug_idx': drug_idx,
                'bs_score': bs_score,
                'tp_score': tp_score,
                'num_visits': len(visits_for_drug),
                'num_patients': len(patients_set)
            })
        
        drug_df = pd.DataFrame(drug_stats)
        
        # 使用中位数划分四象限
        bs_median = drug_df['bs_score'].median()
        tp_median = drug_df['tp_score'].median()
        
        def classify_drug(row):
            if row['tp_score'] >= tp_median and row['bs_score'] >= bs_median:
                return 0  # Chronic, Broad-Spectrum
            elif row['tp_score'] >= tp_median and row['bs_score'] < bs_median:
                return 1  # Chronic, Specialized
            elif row['tp_score'] < tp_median and row['bs_score'] >= bs_median:
                return 2  # Acute, Broad-Spectrum
            else:
                return 3  # Acute, Specialized
        
        drug_df['label'] = drug_df.apply(classify_drug, axis=1)
        
        self.drug_df = drug_df
        labels = torch.tensor(drug_df['label'].values, dtype=torch.long)
        
        label_names = {
            0: 'Chronic, Broad-Spectrum',
            1: 'Chronic, Specialized',
            2: 'Acute, Broad-Spectrum',
            3: 'Acute, Specialized'
        }
        
        print(f"  ✓ Created labels for {len(labels)} drugs")
        print(f"  Label distribution:")
        for label, name in label_names.items():
            count = (drug_df['label'] == label).sum()
            print(f"    {label} ({name}): {count} drugs ({count/len(drug_df)*100:.1f}%)")
        
        return labels, label_names
    
    def create_node_features(self, feature_type='full'):
        """
        为所有节点创建特征
        
        Args:
            feature_type: 'full', 'basic', 'disease_only', 'demo_only'
        
        Returns:
            hetero_data with node features
        """
        print(f"\n[Data Prep] Creating node features (type: {feature_type})...")
        
        # Drug features
        drug_features = []
        for drug_idx in range(self.hetero_data['drug'].num_nodes):
            drug_info = self.drug_df[self.drug_df['drug_idx'] == drug_idx].iloc[0]
            
            if feature_type == 'basic':
                feat = [drug_info['num_visits'], drug_info['num_patients']]
            elif feature_type == 'full':
                feat = [
                    drug_info['num_visits'],
                    drug_info['num_patients'],
                    drug_info['bs_score'],
                    drug_info['tp_score']
                ]
            else:
                feat = [drug_info['num_visits'], drug_info['num_patients']]
            
            drug_features.append(feat)
        
        drug_features = torch.tensor(drug_features, dtype=torch.float)
        
        # Normalize
        scaler = StandardScaler()
        drug_features = torch.from_numpy(
            scaler.fit_transform(drug_features.numpy())
        ).float()
        
        self.hetero_data['drug'].x = drug_features
        
        # Disease features (如果不存在则创建基础特征)
        if not hasattr(self.hetero_data['disease'], 'x') or self.hetero_data['disease'].x is None:
            disease_features = torch.randn(self.hetero_data['disease'].num_nodes, 16)
            self.hetero_data['disease'].x = disease_features
        
        # Patient features
        if not hasattr(self.hetero_data['patient'], 'x') or self.hetero_data['patient'].x is None:
            patient_features = torch.randn(self.hetero_data['patient'].num_nodes, 8)
            self.hetero_data['patient'].x = patient_features
        
        # Visit features (简单创建一个虚拟特征)
        if not hasattr(self.hetero_data['visit'], 'x') or self.hetero_data['visit'].x is None:
            visit_features = torch.randn(self.hetero_data['visit'].num_nodes, 4)
            self.hetero_data['visit'].x = visit_features
        
        # Symptom features
        if not hasattr(self.hetero_data['symptom'], 'x') or self.hetero_data['symptom'].x is None:
            symptom_features = torch.randn(self.hetero_data['symptom'].num_nodes, 8)
            self.hetero_data['symptom'].x = symptom_features
        
        print(f"  ✓ Drug features: {drug_features.shape}")
        print(f"  ✓ Disease features: {self.hetero_data['disease'].x.shape}")
        print(f"  ✓ Patient features: {self.hetero_data['patient'].x.shape}")
        
        return self.hetero_data
    
    def create_train_test_split(self, train_ratio=0.7, val_ratio=0.15, seed=42):
        """
        创建训练/验证/测试集划分
        
        Returns:
            train_mask, val_mask, test_mask
        """
        print(f"\n[Data Prep] Creating train/val/test split...")
        
        num_drugs = self.hetero_data['drug'].num_nodes
        labels = self.drug_df['label'].values
        
        # Stratified split
        indices = np.arange(num_drugs)
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_ratio, stratify=labels, random_state=seed
        )
        
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_ratio_adjusted, 
            stratify=labels[temp_idx], random_state=seed
        )
        
        # Create masks
        train_mask = torch.zeros(num_drugs, dtype=torch.bool)
        val_mask = torch.zeros(num_drugs, dtype=torch.bool)
        test_mask = torch.zeros(num_drugs, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        print(f"  ✓ Train: {train_mask.sum()} drugs ({train_mask.sum()/num_drugs*100:.1f}%)")
        print(f"  ✓ Val:   {val_mask.sum()} drugs ({val_mask.sum()/num_drugs*100:.1f}%)")
        print(f"  ✓ Test:  {test_mask.sum()} drugs ({test_mask.sum()/num_drugs*100:.1f}%)")
        
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        
        return train_mask, val_mask, test_mask
    
    def prepare_full_data(self, feature_type='full'):
        """
        一键准备所有数据
        """
        print("\n" + "="*80)
        print("GNN DATA PREPARATION")
        print("="*80)
        
        labels, label_names = self.create_drug_labels()
        self.hetero_data['drug'].y = labels
        self.label_names = label_names
        
        self.create_node_features(feature_type=feature_type)
        self.create_train_test_split()
        
        print("\n" + "="*80)
        print("DATA PREPARATION COMPLETE")
        print("="*80)
        print(f"\nGraph structure:")
        print(self.hetero_data)
        
        return self.hetero_data, self.train_mask, self.val_mask, self.test_mask


# ==================== 测试 ====================
if __name__ == "__main__":
    from graph_builder import build_ehrshot_graph
    
    data_path = "/home/henry/Desktop/LLM/GraphML/data/"
    graph_builder = build_ehrshot_graph(data_path)
    
    preparator = GNNDataPreparator(graph_builder)
    hetero_data, train_mask, val_mask, test_mask = preparator.prepare_full_data()
    
    print("\n✓ Data preparation test complete!")
