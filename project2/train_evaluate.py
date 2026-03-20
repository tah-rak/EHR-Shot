"""
6_train_evaluate.py
GNN训练和评估模块

包含:
- 统一的训练循环
- 评估指标 (Accuracy, F1, Confusion Matrix)
- 训练曲线可视化
- 结果保存
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


class GNNTrainer:
    """GNN训练器"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
    def train_epoch(self, data, optimizer, mask):
        """训练一个epoch"""
        self.model.train()
        optimizer.zero_grad()
        
        # Forward
        out = self.model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask])
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        pred = out[mask].argmax(dim=1)
        acc = accuracy_score(data.y[mask].cpu(), pred.cpu())
        
        return loss.item(), acc
    
    @torch.no_grad()
    def evaluate(self, data, mask):
        """评估"""
        self.model.eval()
        
        out = self.model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask])
        
        pred = out[mask].argmax(dim=1)
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        return loss.item(), acc, f1, y_true, y_pred
    
    def fit(self, data, train_mask, val_mask, epochs=200, lr=0.01, weight_decay=5e-4, 
            patience=50, verbose=True):
        """
        训练模型
        
        Args:
            data: PyG Data object
            train_mask: 训练集mask
            val_mask: 验证集mask
            epochs: 训练轮数
            lr: 学习率
            weight_decay: L2正则化
            patience: early stopping的patience
            verbose: 是否打印训练信息
        """
        data = data.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_acc = 0
        patience_counter = 0
        
        if verbose:
            pbar = tqdm(range(epochs), desc='Training')
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            # Train
            train_loss, train_acc = self.train_epoch(data, optimizer, train_mask)
            
            # Validate
            val_loss, val_acc, val_f1, _, _ = self.evaluate(data, val_mask)
            
            # History
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}',
                    'val_f1': f'{val_f1:.4f}'
                })
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        self.model.to(self.device)
        
        return self.history
    
    @torch.no_grad()
    def test(self, data, test_mask, label_names=None):
        """测试模型"""
        data = data.to(self.device)
        test_mask = test_mask.to(self.device)
        
        test_loss, test_acc, test_f1, y_true, y_pred = self.evaluate(data, test_mask)
        
        # Classification report
        if label_names is not None:
            target_names = [label_names[i] for i in sorted(label_names.keys())]
        else:
            target_names = None
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        
        results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'report': report
        }
        
        return results
    
    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved training curves to {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, label_names=None, save_path=None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        if label_names is not None:
            labels = [label_names[i] for i in sorted(label_names.keys())]
            labels = [l.split(',')[0] for l in labels]  # Shorten labels
        else:
            labels = None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved confusion matrix to {save_path}")
        
        return fig


def compare_models(results_dict, label_names, save_dir=None):
    """
    比较多个模型的性能
    
    Args:
        results_dict: {model_name: results} 字典
        label_names: 标签名称映射
        save_dir: 保存目录
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # 创建比较表
    comparison_data = []
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Test Acc': f"{results['test_acc']:.4f}",
            'Test F1': f"{results['test_f1']:.4f}",
            'Test Loss': f"{results['test_loss']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))
    
    if save_dir:
        df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
        print(f"\n✓ Saved comparison to {save_dir}/model_comparison.csv")
    
    # 可视化比较
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = [d['Model'] for d in comparison_data]
    accs = [float(d['Test Acc']) for d in comparison_data]
    f1s = [float(d['Test F1']) for d in comparison_data]
    losses = [float(d['Test Loss']) for d in comparison_data]
    
    # Accuracy
    axes[0].bar(models, accs, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Test Accuracy Comparison', fontsize=14)
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # F1 Score
    axes[1].bar(models, f1s, color='coral', alpha=0.8)
    axes[1].set_ylabel('F1 Score (Macro)', fontsize=12)
    axes[1].set_title('Test F1 Score Comparison', fontsize=14)
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # Loss
    axes[2].bar(models, losses, color='mediumseagreen', alpha=0.8)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].set_title('Test Loss Comparison', fontsize=14)
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(losses):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_dir}/model_comparison.png")
    
    plt.show()
    
    return df, fig


# ==================== 测试 ====================
if __name__ == "__main__":
    print("This module should be imported and used with actual data")
