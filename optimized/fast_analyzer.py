"""
optimized/fast_analyzer.py
"""

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms import community
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# ==================== fast analysis ====================
class FastNetworkAnalyzer:
    def __init__(self, graph_builder, top_n_drugs=500):
        """
        快速分析器
        
        Args:
            graph_builder: 图构建器
            top_n_drugs: 只分析前N个药物（按度数排序）
        """
        self.gb = graph_builder
        self.nx_graph = None
        self.top_n_drugs = top_n_drugs
        print(f"\n⚡ Fast Analyzer initialized (analyzing top {top_n_drugs} drugs)")
        
    def convert_to_networkx(self, focus='drug_disease'):
        """将PyG异构图转换为NetworkX图（优化版）"""
        print(f"\nConverting to NetworkX graph (focus: {focus})...")
        
        G = nx.Graph()
        
        if focus == 'drug_disease':
            visit_disease = self.gb.edge_index['visit_disease'].numpy()
            visit_drug = self.gb.edge_index['visit_drug'].numpy()
            
            # 构建映射
            visit_to_diseases = {}
            for i in range(visit_disease.shape[1]):
                visit_idx = visit_disease[0, i]
                disease_idx = visit_disease[1, i]
                if visit_idx not in visit_to_diseases:
                    visit_to_diseases[visit_idx] = []
                visit_to_diseases[visit_idx].append(disease_idx)
            
            visit_to_drugs = {}
            for i in range(visit_drug.shape[1]):
                visit_idx = visit_drug[0, i]
                drug_idx = visit_drug[1, i]
                if visit_idx not in visit_to_drugs:
                    visit_to_drugs[visit_idx] = []
                visit_to_drugs[visit_idx].append(drug_idx)
            
            # 添加所有节点
            for idx, row in self.gb.drug_df.iterrows():
                G.add_node(f"drug_{idx}", 
                          node_type='drug',
                          name=row['drug_name'][:50],
                          num_patients=row['num_patients'],
                          num_diseases=row['num_diseases_treated'],
                          effectiveness=row['effectiveness_score'])
            
            for idx, row in self.gb.disease_df.iterrows():
                G.add_node(f"disease_{idx}",
                          node_type='disease',
                          name=row['disease_name'][:50],
                          num_patients=row['num_patients'])
            
            # 创建Drug-Disease边
            drug_disease_edges = Counter()
            for visit_idx in visit_to_diseases.keys():
                if visit_idx in visit_to_drugs:
                    for disease_idx in visit_to_diseases[visit_idx]:
                        for drug_idx in visit_to_drugs[visit_idx]:
                            drug_disease_edges[(drug_idx, disease_idx)] += 1
            
            # 添加边
            for (drug_idx, disease_idx), weight in drug_disease_edges.items():
                G.add_edge(f"drug_{drug_idx}", f"disease_{disease_idx}", weight=weight)
            
            print(f"✓ Created Drug-Disease network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        self.nx_graph = G
        
        # 🚀 优化1: 提取子图（只包含top drugs）
        print(f"\n⚡ Extracting subgraph with top {self.top_n_drugs} drugs...")
        self._extract_top_drugs_subgraph()
        
        return self
    
    def _extract_top_drugs_subgraph(self):
        """提取包含top drugs的子图"""
        # 计算药物节点的度数
        drug_degrees = {}
        for node in self.nx_graph.nodes():
            if self.nx_graph.nodes[node].get('node_type') == 'drug':
                drug_degrees[node] = self.nx_graph.degree(node)
        
        # 选择top N drugs
        top_drugs = sorted(drug_degrees, key=drug_degrees.get, reverse=True)[:self.top_n_drugs]
        
        # 获取这些drugs连接的所有疾病
        connected_diseases = set()
        for drug in top_drugs:
            for neighbor in self.nx_graph.neighbors(drug):
                if 'disease' in neighbor:
                    connected_diseases.add(neighbor)
        
        # 创建子图
        nodes_to_keep = set(top_drugs) | connected_diseases
        self.full_graph = self.nx_graph.copy()  # 保存完整图
        self.nx_graph = self.nx_graph.subgraph(nodes_to_keep).copy()
        
        print(f"✓ Subgraph created: {self.nx_graph.number_of_nodes()} nodes, {self.nx_graph.number_of_edges()} edges")
        print(f"  (Reduced from {self.full_graph.number_of_nodes()} nodes, {self.full_graph.number_of_edges()} edges)")
    
    def calculate_drug_effectiveness_metrics(self):
        """计算药物有效性指标（快速版）"""
        print("\n⚡ Calculating drug effectiveness metrics (fast mode)...")
        
        drug_metrics = []
        
        # 只计算子图中的药物
        drug_nodes = [n for n in self.nx_graph.nodes() if self.nx_graph.nodes[n].get('node_type') == 'drug']
        
        print(f"  Analyzing {len(drug_nodes)} drugs...")
        
        for node in drug_nodes:
            neighbors = list(self.nx_graph.neighbors(node))
            disease_neighbors = [n for n in neighbors if 'disease' in n]
            
            weighted_degree = sum([self.nx_graph[node][n]['weight'] for n in neighbors])
            avg_weight = weighted_degree / len(neighbors) if len(neighbors) > 0 else 0
            
            drug_data = self.nx_graph.nodes[node]
            
            drug_metrics.append({
                'drug_node': node,
                'drug_name': drug_data.get('name', 'Unknown'),
                'num_diseases_connected': len(disease_neighbors),
                'weighted_degree': weighted_degree,
                'avg_prescription_weight': avg_weight,
                'num_patients': drug_data.get('num_patients', 0),
                'effectiveness_score': drug_data.get('effectiveness', 0)
            })
        
        self.drug_metrics_df = pd.DataFrame(drug_metrics)
        
        # 🚀 优化2: 使用approximate betweenness (快100倍!)
        print("  Computing approximate betweenness centrality...")
        k = min(100, self.nx_graph.number_of_nodes() // 10)  # 采样节点数
        betweenness = nx.betweenness_centrality(self.nx_graph, k=k, weight='weight')
        
        # 🚀 优化3: 使用简化的closeness (只计算drug节点)
        print("  Computing closeness centrality for drugs...")
        closeness = {}
        for node in drug_nodes:
            # 使用ego graph近似
            ego = nx.ego_graph(self.nx_graph, node, radius=2)
            if len(ego) > 1:
                closeness[node] = 1.0 / nx.average_shortest_path_length(ego)
            else:
                closeness[node] = 0
        
        # 添加中心性指标
        self.drug_metrics_df['betweenness'] = self.drug_metrics_df['drug_node'].map(betweenness).fillna(0)
        self.drug_metrics_df['closeness'] = self.drug_metrics_df['drug_node'].map(closeness).fillna(0)
        
        # 综合评分
        scaler = MinMaxScaler()
        metrics_to_scale = ['num_diseases_connected', 'effectiveness_score', 
                           'betweenness', 'closeness']
        
        for col in metrics_to_scale:
            if self.drug_metrics_df[col].std() > 0:
                self.drug_metrics_df[f'{col}_normalized'] = scaler.fit_transform(
                    self.drug_metrics_df[[col]]
                )
            else:
                self.drug_metrics_df[f'{col}_normalized'] = 0
        
        self.drug_metrics_df['composite_score'] = (
            0.3 * self.drug_metrics_df['num_diseases_connected_normalized'] +
            0.3 * self.drug_metrics_df['effectiveness_score_normalized'] +
            0.2 * self.drug_metrics_df['betweenness_normalized'] +
            0.2 * self.drug_metrics_df['closeness_normalized']
        )
        
        print(f"✓ Calculated metrics for {len(self.drug_metrics_df)} drugs")
        
        return self.drug_metrics_df
    
    def detect_communities(self):
        """检测社区（快速版）"""
        print("\n⚡ Detecting communities (fast mode)...")
        
        # 使用label propagation (比Louvain快)
        communities = list(community.label_propagation_communities(self.nx_graph))
        
        node_to_community = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = idx
        
        nx.set_node_attributes(self.nx_graph, node_to_community, 'community')
        
        print(f"✓ Detected {len(communities)} communities")
        
        community_info = []
        for idx, comm in enumerate(communities):
            drugs = [n for n in comm if 'drug' in n]
            diseases = [n for n in comm if 'disease' in n]
            
            community_info.append({
                'community_id': idx,
                'size': len(comm),
                'num_drugs': len(drugs),
                'num_diseases': len(diseases),
                'top_drugs': [self.nx_graph.nodes[d].get('name', 'Unknown') for d in drugs[:3]],
                'top_diseases': [self.nx_graph.nodes[d].get('name', 'Unknown') for d in diseases[:3]]
            })
        
        self.community_df = pd.DataFrame(community_info)
        self.communities = communities
        
        return self.community_df

# ==================== fast visualization ====================
class FastNetworkVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        sns.set_style("whitegrid")
        
    def plot_drug_effectiveness_analysis(self, top_n=20):
        """快速药物有效性分析图"""
        print("\n⚡ Creating effectiveness visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        df = self.analyzer.drug_metrics_df.sort_values('composite_score', ascending=False)
        
        # 1. Top drugs
        top_drugs = df.head(top_n)
        axes[0, 0].barh(range(len(top_drugs)), top_drugs['composite_score'])
        axes[0, 0].set_yticks(range(len(top_drugs)))
        axes[0, 0].set_yticklabels(top_drugs['drug_name'], fontsize=8)
        axes[0, 0].set_xlabel('Composite Effectiveness Score')
        axes[0, 0].set_title(f'Top {top_n} Most Effective Drugs (Fast Analysis)')
        axes[0, 0].invert_yaxis()
        
        # 2. Coverage vs effectiveness
        axes[0, 1].scatter(df['num_diseases_connected'], 
                          df['effectiveness_score'],
                          alpha=0.6, s=50)
        axes[0, 1].set_xlabel('Number of Diseases Treated')
        axes[0, 1].set_ylabel('Effectiveness Score')
        axes[0, 1].set_title('Drug Versatility vs Effectiveness')
        
        # 3. Centrality
        axes[1, 0].scatter(df['betweenness'], df['closeness'],
                          alpha=0.6, c=df['num_diseases_connected'],
                          cmap='viridis', s=100)
        axes[1, 0].set_xlabel('Betweenness Centrality (Approximate)')
        axes[1, 0].set_ylabel('Closeness Centrality')
        axes[1, 0].set_title('Network Centrality Analysis')
        
        # 4. Distribution
        axes[1, 1].hist([df['num_diseases_connected'], 
                        df['weighted_degree']], 
                       bins=30, alpha=0.5, 
                       label=['Diseases', 'Total Prescriptions'])
        axes[1, 1].set_xlabel('Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Drug Usage Distribution')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        print("✓ Effectiveness plot ready")
        return fig
    
    def plot_network_sample(self, sample_size=50):
        """快速网络可视化（减少节点数）"""
        print(f"\n⚡ Creating network visualization ({sample_size} nodes)...")
        
        G = self.analyzer.nx_graph
        
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:sample_size]
        subgraph = G.subgraph(top_nodes)
        
        fig, ax = plt.subplots(figsize=(16, 16))
        
        node_colors = []
        node_sizes = []
        for node in subgraph.nodes():
            if subgraph.nodes[node].get('node_type') == 'drug':
                node_colors.append('lightcoral')
                node_sizes.append(300)
            else:
                node_colors.append('lightblue')
                node_sizes.append(200)
        
        # 使用更快的布局算法
        pos = nx.spring_layout(subgraph, k=1, iterations=30, seed=42)
        
        edges = subgraph.edges()
        weights = [subgraph[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, 
                              width=[min(w/max_weight*3, 3) for w in weights], ax=ax)
        
        nx.draw_networkx_nodes(subgraph, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8, ax=ax)
        
        labels = {}
        for node in list(subgraph.nodes())[:20]:
            name = subgraph.nodes[node].get('name', node)
            labels[node] = name[:15]
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Drug-Disease Network (Fast Analysis - Top {sample_size} nodes)', 
                    fontsize=14)
        ax.axis('off')
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor='lightcoral', markersize=10, label='Drug'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='lightblue', markersize=10, label='Disease')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        print("✓ Network visualization ready")
        return fig
    
    def plot_community_structure(self):
        """快速社区结构图"""
        print("\n⚡ Creating community visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        comm_df = self.analyzer.community_df.sort_values('size', ascending=False)
        axes[0].bar(range(len(comm_df)), comm_df['size'])
        axes[0].set_xlabel('Community ID')
        axes[0].set_ylabel('Community Size')
        axes[0].set_title('Community Size Distribution')
        
        axes[1].scatter(comm_df['num_drugs'], comm_df['num_diseases'],
                       s=comm_df['size']*10, alpha=0.6)
        axes[1].set_xlabel('Number of Drugs')
        axes[1].set_ylabel('Number of Diseases')
        axes[1].set_title('Community Composition')
        
        plt.tight_layout()
        print("✓ Community plot ready")
        return fig

# ==================== fast analysis ====================
def run_fast_analysis(graph_builder, top_n_drugs=500):
    """运行快速分析"""
    
    print("\n" + "="*80)
    print("⚡ FAST NETWORK ANALYSIS MODE")
    print("="*80)
    print(f"Analyzing top {top_n_drugs} drugs (vs all {len(graph_builder.drug_df)} drugs)")
    print("Estimated time: 15-25 minutes (vs 2-3 hours)")
    print("="*80)
    
    analyzer = FastNetworkAnalyzer(graph_builder, top_n_drugs=top_n_drugs)
    analyzer.convert_to_networkx('drug_disease')
    drug_metrics = analyzer.calculate_drug_effectiveness_metrics()
    community_info = analyzer.detect_communities()
    
    visualizer = FastNetworkVisualizer(analyzer)
    

    print("\n" + "="*80)
    print("⚡ FAST ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n📊 Top 10 Most Effective Drugs (from fast analysis):")
    print(drug_metrics.nlargest(10, 'composite_score')[
        ['drug_name', 'num_diseases_connected', 'effectiveness_score', 'composite_score']
    ].to_string(index=False))
    
    print("\n📊 Community Structure:")
    print(community_info.head(5).to_string(index=False))
    
    print("\n" + "="*80)
    
    return analyzer, visualizer

if __name__ == "__main__":
    print("Import this module and use run_fast_analysis(graph_builder)")