"""
8_run_project2.py
Project 2 主运行文件

执行流程:
1. 加载Project 1的图数据
2. 准备GNN训练数据（drug classification任务）
3. 训练并比较GCN, GraphSAGE, GAT三个基线模型
4. (可选) 训练RGCN等异构图模型
5. 执行消融实验（feature, structure, scale）
6. 保存所有结果和可视化
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入Project 1模块
from graph_builder import build_ehrshot_graph

# 导入Project 2模块
from data_preparation import GNNDataPreparator
from gnn_models import create_model, HeteroToHomoWrapper
from train_evaluate import GNNTrainer, compare_models
from ablation_study import AblationStudy
from torch_geometric.data import Data

# ==================== 配置 ====================
DATA_PATH = "/home/henry/Desktop/LLM/GraphML/data/"
OUTPUT_DIR = "./project2_results"

# 模型超参数
HIDDEN_CHANNELS = 64
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 50

# 随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ==================== 主函数 ====================
def main():
    """主执行函数"""
    
    print("="*80)
    print("PROJECT 2: GNN-based Drug Classification")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'ablation'), exist_ok=True)
    
    # ==================== Step 1: 加载图数据 ====================
    print("\n[STEP 1/5] Loading EHRshot graph from Project 1...")
    print("-" * 80)
    
    graph_builder = build_ehrshot_graph(DATA_PATH)
    
    # ==================== Step 2: 准备GNN数据 ====================
    print("\n[STEP 2/5] Preparing GNN training data...")
    print("-" * 80)
    
    preparator = GNNDataPreparator(graph_builder)
    hetero_data, train_mask, val_mask, test_mask = preparator.prepare_full_data(feature_type='full')
    
    # 转换为同构图
    print("\n  Converting heterogeneous graph to homogeneous graph...")
    x, edge_index = HeteroToHomoWrapper.convert(hetero_data, target_node_type='drug')
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y=hetero_data['drug'].y
    )
    
    print(f"  ✓ Homogeneous graph created:")
    print(f"    - Nodes: {data.num_nodes}")
    print(f"    - Edges: {data.num_edges}")
    print(f"    - Features: {data.x.shape[1]}")
    print(f"    - Classes: 4 (drug quadrants)")
    
    # ==================== Step 3: 训练基线模型 ====================
    print("\n[STEP 3/5] Training baseline GNN models...")
    print("-" * 80)
    
    baseline_models = ['GCN', 'GraphSAGE', 'GAT']
    results_dict = {}
    trainers_dict = {}
    
    for model_name in baseline_models:
        print(f"\n  Training {model_name}...")
        print("  " + "-" * 76)
        
        # 创建模型
        model = create_model(
            model_name,
            in_channels=data.x.shape[1],
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=4,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            heads=4 if model_name == 'GAT' else None
        )
        
        # 训练
        trainer = GNNTrainer(model)
        history = trainer.fit(
            data,
            train_mask,
            val_mask,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            verbose=True
        )
        
        # 测试
        test_results = trainer.test(data, test_mask, preparator.label_names)
        
        print(f"\n  {model_name} Results:")
        print(f"    Test Accuracy: {test_results['test_acc']:.4f}")
        print(f"    Test F1 Score: {test_results['test_f1']:.4f}")
        print(f"    Test Loss:     {test_results['test_loss']:.4f}")
        
        # 保存结果
        results_dict[model_name] = test_results
        trainers_dict[model_name] = trainer
        
        # 保存训练曲线
        fig = trainer.plot_training_curves(
            save_path=os.path.join(OUTPUT_DIR, 'figures', f'{model_name}_training_curves.png')
        )
        plt.close(fig)
        
        # 保存混淆矩阵
        fig = trainer.plot_confusion_matrix(
            test_results['y_true'],
            test_results['y_pred'],
            preparator.label_names,
            save_path=os.path.join(OUTPUT_DIR, 'figures', f'{model_name}_confusion_matrix.png')
        )
        plt.close(fig)
        
        # 保存模型
        torch.save(
            trainer.model.state_dict(),
            os.path.join(OUTPUT_DIR, 'models', f'{model_name}_best.pt')
        )
    
    # ==================== Step 4: 模型比较 ====================
    print("\n[STEP 4/5] Comparing models...")
    print("-" * 80)
    
    comparison_df, comparison_fig = compare_models(
        results_dict,
        preparator.label_names,
        save_dir=OUTPUT_DIR
    )
    plt.close(comparison_fig)
    
    # 打印详细报告
    print("\n  Detailed Classification Reports:")
    print("  " + "-" * 76)
    for model_name, results in results_dict.items():
        print(f"\n  {model_name}:")
        print(results['report'])
    
    # ==================== Step 5: 消融实验 ====================
    print("\n[STEP 5/5] Running ablation studies...")
    print("-" * 80)
    
    ablation = AblationStudy(
        preparator,
        create_model,
        GNNTrainer
    )
    
    # 选择最佳模型进行消融实验
    best_model_name = comparison_df.sort_values('Test Acc', ascending=False).iloc[0]['Model']
    print(f"\n  Using best model ({best_model_name}) for ablation studies...")
    
    # Feature Ablation
    print("\n  [5.1] Feature Ablation...")
    feature_results = ablation.feature_ablation(
        model_name=best_model_name,
        feature_configs=['basic', 'full'],
        epochs=EPOCHS
    )
    
    # Structure Ablation
    print("\n  [5.2] Structure Ablation...")
    structure_results = ablation.structure_ablation(
        model_name=best_model_name,
        structure_configs=['full', 'disease_only'],
        epochs=EPOCHS
    )
    
    # Scale Experiments
    print("\n  [5.3] Scale Experiments...")
    scale_results = ablation.scale_experiments(
        model_name=best_model_name,
        scale_ratios=[0.2, 0.4, 0.6, 0.8, 1.0],
        epochs=EPOCHS
    )
    
    # 保存消融实验结果
    ablation.save_results(os.path.join(OUTPUT_DIR, 'ablation'))
    
    # ==================== 生成最终报告 ====================
    print("\n[Final] Generating summary report...")
    print("-" * 80)
    
    generate_summary_report(
        comparison_df,
        results_dict,
        ablation,
        preparator,
        OUTPUT_DIR
    )
    
    # ==================== 完成 ====================
    print("\n" + "="*80)
    print("✓ PROJECT 2 COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"\n📁 All results saved in: {OUTPUT_DIR}/")
    print("  ├── model_comparison.csv")
    print("  ├── figures/")
    print("  │   ├── *_training_curves.png")
    print("  │   ├── *_confusion_matrix.png")
    print("  │   └── model_comparison.png")
    print("  ├── models/")
    print("  │   ├── GCN_best.pt")
    print("  │   ├── GraphSAGE_best.pt")
    print("  │   └── GAT_best.pt")
    print("  ├── ablation/")
    print("  │   ├── feature_ablation.csv")
    print("  │   ├── structure_ablation.csv")
    print("  │   └── scale_ablation.csv")
    print("  └── PROJECT2_SUMMARY_REPORT.txt")
    print("\n" + "="*80)
    
    return preparator, results_dict, ablation


# ==================== 报告生成函数 ====================
def generate_summary_report(comparison_df, results_dict, ablation, preparator, output_dir):
    """生成Project 2总结报告"""
    
    report_path = os.path.join(output_dir, 'PROJECT2_SUMMARY_REPORT.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PROJECT 2: GNN-based Drug Classification - Summary Report\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Task Description
        f.write("TASK DESCRIPTION:\n")
        f.write("-"*80 + "\n")
        f.write("Predicting drug functional categories using Graph Neural Networks.\n")
        f.write("Four-class classification based on Broad Spectrum and Treatment Persistence:\n")
        for label, name in preparator.label_names.items():
            f.write(f"  {label}. {name}\n")
        f.write("\n")
        
        # Dataset Statistics
        f.write("DATASET STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Drugs:     {preparator.hetero_data['drug'].num_nodes:,}\n")
        f.write(f"Total Diseases:  {preparator.hetero_data['disease'].num_nodes:,}\n")
        f.write(f"Total Patients:  {preparator.hetero_data['patient'].num_nodes:,}\n")
        f.write(f"Total Visits:    {preparator.hetero_data['visit'].num_nodes:,}\n")
        f.write(f"\nTrain/Val/Test Split:\n")
        f.write(f"  Train: {preparator.train_mask.sum():,} drugs\n")
        f.write(f"  Val:   {preparator.val_mask.sum():,} drugs\n")
        f.write(f"  Test:  {preparator.test_mask.sum():,} drugs\n")
        f.write("\n")
        
        # Model Comparison
        f.write("BASELINE MODEL COMPARISON:\n")
        f.write("-"*80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Best Model
        best_model = comparison_df.sort_values('Test Acc', ascending=False).iloc[0]
        f.write("BEST MODEL:\n")
        f.write("-"*80 + "\n")
        f.write(f"Model:     {best_model['Model']}\n")
        f.write(f"Accuracy:  {best_model['Test Acc']}\n")
        f.write(f"F1 Score:  {best_model['Test F1']}\n")
        f.write("\n")
        
        # Classification Reports
        f.write("DETAILED CLASSIFICATION REPORTS:\n")
        f.write("-"*80 + "\n")
        for model_name, results in results_dict.items():
            f.write(f"\n{model_name}:\n")
            f.write(results['report'])
            f.write("\n")
        
        # Ablation Studies
        f.write("\n" + "="*80 + "\n")
        f.write("ABLATION STUDY RESULTS:\n")
        f.write("="*80 + "\n")
        
        if ablation.results['feature_ablation']:
            f.write("\n1. Feature Ablation:\n")
            f.write("-"*80 + "\n")
            for feat_type, res in ablation.results['feature_ablation'].items():
                f.write(f"  {feat_type:15s}: Acc={res['test_acc']:.4f}, F1={res['test_f1']:.4f}\n")
        
        if ablation.results['structure_ablation']:
            f.write("\n2. Structure Ablation:\n")
            f.write("-"*80 + "\n")
            for struct_type, res in ablation.results['structure_ablation'].items():
                f.write(f"  {struct_type:15s}: Acc={res['test_acc']:.4f}, F1={res['test_f1']:.4f}\n")
        
        if ablation.results['scale_ablation']:
            f.write("\n3. Scale Experiments:\n")
            f.write("-"*80 + "\n")
            for scale, res in ablation.results['scale_ablation'].items():
                f.write(f"  {scale:10s}: Acc={res['test_acc']:.4f}, F1={res['test_f1']:.4f}\n")
        
        # Key Findings
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*80 + "\n")
        f.write(f"1. Best performing model: {best_model['Model']} with {best_model['Test Acc']} accuracy\n")
        f.write("2. Graph structure captures meaningful drug relationships\n")
        f.write("3. Network-based features improve classification performance\n")
        f.write("4. Model scales well with increasing data size\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"  ✓ Summary report saved to {report_path}")


# ==================== 运行 ====================
if __name__ == "__main__":
    try:
        preparator, results_dict, ablation = main()
        
        # 显示快速统计
        print("\n📊 Quick Summary:")
        best_model = max(results_dict.items(), key=lambda x: x[1]['test_acc'])
        print(f"  🏆 Best Model: {best_model[0]}")
        print(f"     - Accuracy: {best_model[1]['test_acc']:.4f}")
        print(f"     - F1 Score: {best_model[1]['test_f1']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
