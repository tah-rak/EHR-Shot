# Project 2: GNN-based Drug Classification

## 📋 Overview

This project extends Project 1 by building Graph Neural Network (GNN) models to predict drug functional categories. The task is a **4-class node classification** problem that classifies drugs into four quadrants based on their clinical usage patterns.

### Prediction Task: Drug Classification

**Four Categories:**
1. **Chronic, Broad-Spectrum** - Drugs used repeatedly across many diseases
2. **Chronic, Specialized** - Drugs used repeatedly for specific diseases  
3. **Acute, Broad-Spectrum** - Single-use drugs across many diseases
4. **Acute, Specialized** - Single-use drugs for specific diseases

**Clinical Significance:** Understanding drug functional categories helps in:
- Treatment planning and protocol design
- Drug repurposing research
- Clinical decision support systems
- Pharmacy inventory management

---

## 🗂️ Project Structure

```
project2/
├── 4_data_preparation.py      # Convert heterogeneous graph to GNN data
├── 5_gnn_models.py             # GNN model definitions (GCN, GraphSAGE, GAT, RGCN)
├── 6_train_evaluate.py         # Training loops and evaluation metrics
├── 7_ablation_study.py         # Ablation experiments
├── 8_run_project2.py           # Main execution script
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

**Required Libraries:**
- `torch >= 2.0.0`
- `torch-geometric >= 2.3.0`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tqdm`

### Step-by-Step Execution

**1. Prepare Your Data**

Make sure you have completed Project 1 and have the following from `graph_builder.py`:
- `build_ehrshot_graph()` function
- EHRshot dataset in `/home/henry/Desktop/LLM/GraphML/data/`

**2. Run the Complete Pipeline**

```bash
python 8_run_project2.py
```

This will:
- Load the heterogeneous graph from Project 1
- Create drug classification labels (4 categories)
- Train 3 baseline GNN models (GCN, GraphSAGE, GAT)
- Compare model performance
- Run ablation experiments
- Save all results and visualizations

**3. Check Results**

Results will be saved in `./project2_results/`:
```
project2_results/
├── model_comparison.csv
├── PROJECT2_SUMMARY_REPORT.txt
├── figures/
│   ├── GCN_training_curves.png
│   ├── GCN_confusion_matrix.png
│   ├── GraphSAGE_training_curves.png
│   ├── GraphSAGE_confusion_matrix.png
│   ├── GAT_training_curves.png
│   ├── GAT_confusion_matrix.png
│   └── model_comparison.png
├── models/
│   ├── GCN_best.pt
│   ├── GraphSAGE_best.pt
│   └── GAT_best.pt
└── ablation/
    ├── feature_ablation.csv
    ├── structure_ablation.csv
    └── scale_ablation.csv
```

---

## 📊 Detailed Module Usage

### Module 1: Data Preparation

```python
from graph_builder import build_ehrshot_graph
from data_preparation import GNNDataPreparator

# Load Project 1 graph
graph_builder = build_ehrshot_graph("/path/to/data/")

# Prepare GNN data
preparator = GNNDataPreparator(graph_builder)
hetero_data, train_mask, val_mask, test_mask = preparator.prepare_full_data(
    feature_type='full'  # Options: 'basic', 'full'
)
```

**Key Functions:**
- `create_drug_labels()` - Generate 4-class labels from BS and TP scores
- `create_node_features()` - Create node features for all node types
- `create_train_test_split()` - Stratified train/val/test split

### Module 2: GNN Models

```python
from gnn_models import create_model, HeteroToHomoWrapper

# Convert heterogeneous to homogeneous graph
x, edge_index = HeteroToHomoWrapper.convert(hetero_data, target_node_type='drug')

# Create model
model = create_model(
    'GCN',  # Options: 'GCN', 'GraphSAGE', 'GAT', 'RGCN'
    in_channels=4,
    hidden_channels=64,
    out_channels=4,
    num_layers=2,
    dropout=0.5
)
```

**Supported Models:**
- **GCN** (Graph Convolutional Network) - Basic spectral convolution
- **GraphSAGE** - Inductive learning with neighborhood sampling
- **GAT** (Graph Attention Network) - Attention-based aggregation
- **RGCN** (Relational GCN) - For heterogeneous graphs with multiple edge types

### Module 3: Training & Evaluation

```python
from train_evaluate import GNNTrainer, compare_models
from torch_geometric.data import Data

# Create data object
data = Data(x=x, edge_index=edge_index, y=labels)

# Train
trainer = GNNTrainer(model, device='cuda')
history = trainer.fit(
    data, train_mask, val_mask,
    epochs=200,
    lr=0.01,
    patience=50
)

# Test
results = trainer.test(data, test_mask, label_names)

# Visualize
trainer.plot_training_curves(save_path='training_curves.png')
trainer.plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png')
```

**Evaluation Metrics:**
- Accuracy
- F1 Score (macro)
- Classification Report
- Confusion Matrix

### Module 4: Ablation Studies

```python
from ablation_study import AblationStudy

ablation = AblationStudy(preparator, create_model, GNNTrainer)

# Feature ablation
feature_results = ablation.feature_ablation(
    model_name='GCN',
    feature_configs=['basic', 'full']
)

# Structure ablation
structure_results = ablation.structure_ablation(
    model_name='GCN',
    structure_configs=['full', 'disease_only']
)

# Scale experiments
scale_results = ablation.scale_experiments(
    model_name='GCN',
    scale_ratios=[0.2, 0.4, 0.6, 0.8, 1.0]
)
```

---

## 🔬 Experimental Design

### Baseline Models (Required)

All three models are trained and evaluated on:
- **Same graph structure**
- **Same node features**
- **Same train/val/test split**
- **Same hyperparameters** (where applicable)

This ensures fair comparison.

### Ablation Experiments

**1. Feature Ablation**
- `basic`: Only num_visits and num_patients
- `full`: Includes BS score and TP score

**2. Structure Ablation**
- `full`: Complete heterogeneous graph
- `disease_only`: Only drug-visit-disease paths
- `no_visit`: Direct drug-disease connections

**3. Scale Experiments**
- Train on 20%, 40%, 60%, 80%, 100% of data
- Analyze generalization vs data size

---

## 📈 Expected Results

Based on the graph structure and task design, you should expect:

**Model Performance:**
- Accuracy: ~60-80% (depending on data quality)
- F1 Score: ~0.55-0.75 (macro)
- GAT typically performs best due to attention mechanism
- GraphSAGE is more efficient for large graphs

**Ablation Insights:**
- Full features > basic features (BS and TP are informative)
- Full graph > simplified structures (more context helps)
- Performance saturates around 60-80% data (diminishing returns)

**Common Issues:**
- Class imbalance may affect some quadrants
- Sparse connections for rare drugs
- May need more epochs for GAT convergence

---

## 🎓 For Course Submission

### Report Sections (refer to Project Description PDF)

Your report should include:

1. **Introduction** (~1 page)
   - Motivation and clinical significance
   - Research question
   - Related work (7+ citations)

2. **Graph Construction** (~1-1.5 pages)
   - Dataset preprocessing
   - Node types and features
   - Edge definition
   - Graph statistics

3. **Model Details** (~1 page)
   - Task definition (4-class node classification)
   - GNN architectures (GCN, GraphSAGE, GAT)
   - Hyperparameters
   - Training setup

4. **Results** (~2 pages)
   - Performance tables
   - Training curves
   - Confusion matrices
   - Statistical significance tests (if applicable)

5. **Experimental Analysis** (~1-1.5 pages)
   - Feature ablation findings
   - Structure ablation findings
   - Scale experiment findings
   - Clinical interpretation

6. **Conclusion** (~0.5 page)
   - Key findings
   - Limitations
   - Future directions

### Key Figures to Include

- Model comparison bar chart
- Training/validation curves for best model
- Confusion matrix for best model
- Ablation study results
- Example predictions with explanations

---

## 🔧 Customization

### Change Hyperparameters

Edit `8_run_project2.py`:
```python
HIDDEN_CHANNELS = 128  # Default: 64
NUM_LAYERS = 3         # Default: 2
DROPOUT = 0.3          # Default: 0.5
LEARNING_RATE = 0.001  # Default: 0.01
```

### Add More Models

In `5_gnn_models.py`, implement new model class:
```python
class MyCustomGNN(BaseGNN):
    def __init__(self, ...):
        # Your implementation
        pass
    
    def forward(self, x, edge_index):
        # Your forward pass
        pass
```

Then add to `create_model()` factory function.

### Modify Label Creation

In `4_data_preparation.py`, change `create_drug_labels()`:
```python
def classify_drug(row):
    # Your custom classification logic
    if custom_condition:
        return 0
    # ...
```

---

## 💡 Tips & Best Practices

**1. Start Simple**
- Run with default hyperparameters first
- Use small subset for debugging
- Check data preparation carefully

**2. Computational Resources**
- GCN is fastest (~2-3 min per model)
- GAT is slowest (~5-10 min per model)
- Use GPU if available (set `device='cuda'`)

**3. Debugging**
- Check train/val/test split is stratified
- Verify edge_index shape is [2, num_edges]
- Ensure no data leakage between splits

**4. Improving Performance**
- Try different hidden_channels (32, 64, 128)
- Adjust learning rate (0.001, 0.01, 0.1)
- Increase patience for early stopping
- Use learning rate scheduler

**5. Clinical Interpretation**
- Don't just report numbers
- Explain why certain drugs are misclassified
- Connect findings to real clinical patterns
- Discuss practical implications

---

## 📚 References

**Key Papers:**
- Kipf & Welling (2017) - Semi-Supervised Classification with GCNs
- Hamilton et al. (2017) - Inductive Representation Learning on Large Graphs (GraphSAGE)
- Veličković et al. (2018) - Graph Attention Networks
- Schlichtkrull et al. (2018) - Modeling Relational Data with GCNs (RGCN)

**EHR & Healthcare:**
- Choi et al. (2020) - Learning the Graphical Structure of EHR Data
- Bauer-Mehren et al. (2013) - Network Analysis of EHR for Clinical Research

---

## ❓ Troubleshooting

**Q: ModuleNotFoundError: No module named 'torch_geometric'**  
A: Install PyTorch Geometric: `pip install torch-geometric`

**Q: CUDA out of memory**  
A: Reduce batch size, hidden_channels, or use CPU: `device='cpu'`

**Q: Model accuracy is very low (<40%)**  
A: Check label distribution, increase epochs, adjust learning rate

**Q: Training is too slow**  
A: Use GraphSAGE instead of GAT, reduce num_layers, use smaller hidden_channels

**Q: How to reproduce exact results?**  
A: Set all random seeds:
```python
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
```

---

## 📧 Contact & Support

For questions or issues:
1. Check the Project Description PDF
2. Review course materials on Canvas
3. Ask during office hours
4. Email instructor/TA

---

## ✅ Checklist for Submission

- [ ] All code runs without errors
- [ ] Results saved in `project2_results/`
- [ ] Report follows required structure (6-8 pages)
- [ ] All 3 baseline models implemented and compared
- [ ] At least 3 ablation experiments completed
- [ ] Figures are clear and labeled
- [ ] Tables are formatted properly
- [ ] Citations are included
- [ ] Code is well-commented
- [ ] README explains how to run code

---

**Good luck with your project! 🚀**
