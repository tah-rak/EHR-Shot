"""
QUICK_START_EXAMPLE.py
快速开始示例 - 演示如何使用各个模块

这个脚本展示了如何单独使用每个模块进行GNN训练
"""

import torch
import numpy as np
from torch_geometric.data import Data

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*80)
print("Project 2: Quick Start Example")
print("="*80)

# ==================== Step 1: 数据准备 ====================
print("\n[Step 1] Loading and preparing data...")

from graph_builder import build_ehrshot_graph
from data_preparation import GNNDataPreparator
from gnn_models import HeteroToHomoWrapper

# 加载Project 1的图
DATA_PATH = "/home/henry/Desktop/LLM/GraphML/data/"
graph_builder = build_ehrshot_graph(DATA_PATH)

# 准备GNN数据
preparator = GNNDataPreparator(graph_builder)
hetero_data, train_mask, val_mask, test_mask = preparator.prepare_full_data(feature_type='full')

# 转换为同构图
x, edge_index = HeteroToHomoWrapper.convert(hetero_data, target_node_type='drug')
data = Data(x=x, edge_index=edge_index, y=hetero_data['drug'].y)

print(f"✓ Data prepared:")
print(f"  - {data.num_nodes} nodes")
print(f"  - {data.num_edges} edges")
print(f"  - {data.x.shape[1]} features per node")
print(f"  - 4 classes (drug quadrants)")

# ==================== Step 2: 创建模型 ====================
print("\n[Step 2] Creating GNN model...")

from gnn_models import create_model

model = create_model(
    'GCN',  # 可以改为 'GraphSAGE' 或 'GAT'
    in_channels=data.x.shape[1],
    hidden_channels=64,
    out_channels=4,
    num_layers=2,
    dropout=0.5
)

print(f"✓ Model created: GCN")
print(f"  - Hidden channels: 64")
print(f"  - Num layers: 2")
print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ==================== Step 3: 训练模型 ====================
print("\n[Step 3] Training model...")

from train_evaluate import GNNTrainer

trainer = GNNTrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')

# 训练（使用较少的epochs进行快速演示）
history = trainer.fit(
    data,
    train_mask,
    val_mask,
    epochs=50,  # 正式训练时使用200
    lr=0.01,
    patience=20,
    verbose=True
)

print(f"\n✓ Training complete!")
print(f"  - Final train acc: {history['train_acc'][-1]:.4f}")
print(f"  - Final val acc: {history['val_acc'][-1]:.4f}")

# ==================== Step 4: 测试模型 ====================
print("\n[Step 4] Testing model...")

test_results = trainer.test(data, test_mask, preparator.label_names)

print(f"\n✓ Test Results:")
print(f"  - Accuracy: {test_results['test_acc']:.4f}")
print(f"  - F1 Score: {test_results['test_f1']:.4f}")
print(f"  - Loss:     {test_results['test_loss']:.4f}")

print("\n" + "-"*80)
print("Classification Report:")
print(test_results['report'])

# ==================== Step 5: 可视化 ====================
print("\n[Step 5] Creating visualizations...")

# 训练曲线
fig1 = trainer.plot_training_curves(save_path='quick_start_training_curves.png')
print("✓ Saved training curves to quick_start_training_curves.png")

# 混淆矩阵
fig2 = trainer.plot_confusion_matrix(
    test_results['y_true'],
    test_results['y_pred'],
    preparator.label_names,
    save_path='quick_start_confusion_matrix.png'
)
print("✓ Saved confusion matrix to quick_start_confusion_matrix.png")

# ==================== 完成 ====================
print("\n" + "="*80)
print("✓ Quick start example complete!")
print("="*80)
print("\nNext steps:")
print("  1. Run the full pipeline: python 8_run_project2.py")
print("  2. Compare multiple models (GCN, GraphSAGE, GAT)")
print("  3. Perform ablation studies")
print("  4. Write your project report")
print("\n" + "="*80)


# ==================== 额外示例: 比较多个模型 ====================
def compare_multiple_models_example():
    """
    示例: 如何快速比较多个模型
    """
    from train_evaluate import compare_models
    
    print("\n[Bonus] Comparing multiple models...")
    
    results_dict = {}
    
    for model_name in ['GCN', 'GraphSAGE', 'GAT']:
        print(f"\nTraining {model_name}...")
        
        model = create_model(
            model_name,
            in_channels=data.x.shape[1],
            hidden_channels=64,
            out_channels=4
        )
        
        trainer = GNNTrainer(model)
        trainer.fit(data, train_mask, val_mask, epochs=30, verbose=False)
        test_results = trainer.test(data, test_mask, preparator.label_names)
        
        results_dict[model_name] = test_results
        print(f"  {model_name} Test Acc: {test_results['test_acc']:.4f}")
    
    # 比较结果
    comparison_df, comparison_fig = compare_models(
        results_dict,
        preparator.label_names,
        save_dir='.'
    )
    
    print("\n✓ Model comparison complete!")
    return results_dict


# ==================== 额外示例: 消融实验 ====================
def ablation_study_example():
    """
    示例: 如何进行特征消融实验
    """
    from ablation_study import AblationStudy
    
    print("\n[Bonus] Running feature ablation study...")
    
    ablation = AblationStudy(preparator, create_model, GNNTrainer)
    
    # 特征消融
    feature_results = ablation.feature_ablation(
        model_name='GCN',
        feature_configs=['basic', 'full'],
        epochs=30
    )
    
    print("\n✓ Feature ablation complete!")
    print("\nResults:")
    for feat_type, res in feature_results.items():
        print(f"  {feat_type}: Acc={res['test_acc']:.4f}, F1={res['test_f1']:.4f}")
    
    return feature_results


# 取消注释以运行额外示例:
# compare_multiple_models_example()
# ablation_study_example()
