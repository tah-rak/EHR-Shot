"""
2_network_analyzer.py
网络分析与可视化模块

使用方法:
    from network_analyzer import NetworkAnalyzer, NetworkVisualizer
    analyzer = NetworkAnalyzer(graph_builder)
    analyzer.convert_to_networkx('drug_disease')
    visualizer = NetworkVisualizer(analyzer)
"""

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms import community
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

# ==================== 网络分析器 ====================
class NetworkAnalyzer:
    def __init__(self, graph_builder):
        self.gb = graph_builder
        self.nx_graph = None
        
    def convert_to_networkx(self, focus='drug_disease'):
        """将PyG异构图转换为NetworkX图"""
        print(f"\nConverting to NetworkX graph (focus: {focus})...")
        
        G = nx.Graph()
        
        if focus == 'drug_disease':
            # Drug-Disease二部图（通过Visit连接）
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
            
            # 添加节点
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
        return self
    
    def calculate_drug_effectiveness_metrics(self):
        """计算药物有效性的网络指标"""
        print("\nCalculating drug effectiveness metrics...")
        
        if self.nx_graph is None:
            self.convert_to_networkx('drug_disease')
        
        drug_metrics = []
        
        for node in self.nx_graph.nodes():
            if self.nx_graph.nodes[node].get('node_type') == 'drug':
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
                    'effectiveness_score': drug_data.get('effectiveness', 0),
                    'betweenness': nx.betweenness_centrality(self.nx_graph, weight='weight').get(node, 0),
                    'closeness': nx.closeness_centrality(self.nx_graph, distance='weight').get(node, 0)
                })
        
        self.drug_metrics_df = pd.DataFrame(drug_metrics)
        
        # 综合评分
        scaler = MinMaxScaler()
        metrics_to_scale = ['num_diseases_connected', 'effectiveness_score', 
                           'betweenness', 'closeness']
        
        for col in metrics_to_scale:
            self.drug_metrics_df[f'{col}_normalized'] = scaler.fit_transform(
                self.drug_metrics_df[[col]]
            )
        
        self.drug_metrics_df['composite_score'] = (
            0.3 * self.drug_metrics_df['num_diseases_connected_normalized'] +
            0.3 * self.drug_metrics_df['effectiveness_score_normalized'] +
            0.2 * self.drug_metrics_df['betweenness_normalized'] +
            0.2 * self.drug_metrics_df['closeness_normalized']
        )
        
        print(f"✓ Calculated metrics for {len(self.drug_metrics_df)} drugs")
        
        return self.drug_metrics_df
    
    def detect_communities(self):
        """检测疾病-药物社区"""
        print("\nDetecting communities...")
        
        if self.nx_graph is None:
            self.convert_to_networkx('drug_disease')
        
        communities = community.greedy_modularity_communities(self.nx_graph, weight='weight')
        
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

# ==================== 可视化器 ====================
class NetworkVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        sns.set_style("whitegrid")
        
    def plot_drug_effectiveness_analysis(self, top_n=20):
        """可视化药物有效性分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        df = self.analyzer.drug_metrics_df.sort_values('composite_score', ascending=False)
        
        # 1. Top drugs
        top_drugs = df.head(top_n)
        axes[0, 0].barh(range(len(top_drugs)), top_drugs['composite_score'])
        axes[0, 0].set_yticks(range(len(top_drugs)))
        axes[0, 0].set_yticklabels(top_drugs['drug_name'], fontsize=8)
        axes[0, 0].set_xlabel('Composite Effectiveness Score')
        axes[0, 0].set_title(f'Top {top_n} Most Effective Drugs')
        axes[0, 0].invert_yaxis()
        
        # 2. Coverage vs effectiveness
        axes[0, 1].scatter(df['num_diseases_connected'], 
                          df['effectiveness_score'],
                          alpha=0.6, s=df['num_patients']/10)
        axes[0, 1].set_xlabel('Number of Diseases Treated')
        axes[0, 1].set_ylabel('Effectiveness Score')
        axes[0, 1].set_title('Drug Versatility vs Effectiveness')
        
        top_5 = df.head(5)
        for idx, row in top_5.iterrows():
            axes[0, 1].annotate(row['drug_name'][:20], 
                               (row['num_diseases_connected'], row['effectiveness_score']),
                               fontsize=7, alpha=0.7)
        
        # 3. Centrality
        axes[1, 0].scatter(df['betweenness'], df['closeness'],
                          alpha=0.6, c=df['num_diseases_connected'],
                          cmap='viridis', s=100)
        axes[1, 0].set_xlabel('Betweenness Centrality')
        axes[1, 0].set_ylabel('Closeness Centrality')
        axes[1, 0].set_title('Network Centrality Analysis')
        plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Disease Coverage')
        
        # 4. Distribution
        axes[1, 1].hist([df['num_diseases_connected'], 
                        df['num_patients'],
                        df['weighted_degree']], 
                       bins=30, alpha=0.5, 
                       label=['Diseases', 'Patients', 'Total Prescriptions'])
        axes[1, 1].set_xlabel('Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Drug Usage Distribution')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        print("✓ Drug effectiveness analysis plot ready")
        plt.show()
        
        return fig
    
    def plot_network_sample(self, sample_size=100):
        """绘制网络样本"""
        print(f"\nVisualizing network sample ({sample_size} nodes)...")
        
        G = self.analyzer.nx_graph
        
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:sample_size]
        subgraph = G.subgraph(top_nodes)
        
        fig, ax = plt.subplots(figsize=(20, 20))
        
        node_colors = []
        node_sizes = []
        for node in subgraph.nodes():
            if subgraph.nodes[node].get('node_type') == 'drug':
                node_colors.append('lightcoral')
                node_sizes.append(300)
            else:
                node_colors.append('lightblue')
                node_sizes.append(200)
        
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        edges = subgraph.edges()
        weights = [subgraph[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, 
                              width=[w/max_weight*3 for w in weights], ax=ax)
        
        nx.draw_networkx_nodes(subgraph, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8, ax=ax)
        
        labels = {}
        for node in list(subgraph.nodes())[:30]:
            name = subgraph.nodes[node].get('name', node)
            labels[node] = name[:15]
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Drug-Disease Network Sample (Top {sample_size} nodes)', fontsize=16)
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
        plt.show()
        
        return fig
    
    def plot_community_structure(self):
        """可视化社区结构"""
        if not hasattr(self.analyzer, 'communities'):
            self.analyzer.detect_communities()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
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
        
        for idx, row in comm_df.head(5).iterrows():
            axes[1].annotate(f"C{row['community_id']}", 
                           (row['num_drugs'], row['num_diseases']), fontsize=10)
        
        plt.tight_layout()
        print("✓ Community structure plot ready")
        plt.show()
        
        return fig

# ==================== 完整分析流程 ====================
def run_full_analysis(graph_builder):
    """运行完整的网络分析"""
    
    analyzer = NetworkAnalyzer(graph_builder)
    analyzer.convert_to_networkx('drug_disease')
    drug_metrics = analyzer.calculate_drug_effectiveness_metrics()
    community_info = analyzer.detect_communities()
    
    visualizer = NetworkVisualizer(analyzer)
    visualizer.plot_drug_effectiveness_analysis(top_n=20)
    visualizer.plot_network_sample(sample_size=100)
    visualizer.plot_community_structure()
    
    # 总结报告
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n📊 Top 10 Most Effective Drugs:")
    print(drug_metrics.nlargest(10, 'composite_score')[
        ['drug_name', 'num_diseases_connected', 'effectiveness_score', 'composite_score']
    ].to_string(index=False))
    
    print("\n📊 Top 10 Most Versatile Drugs:")
    print(drug_metrics.nlargest(10, 'num_diseases_connected')[
        ['drug_name', 'num_diseases_connected', 'num_patients', 'effectiveness_score']
    ].to_string(index=False))
    
    print("\n📊 Community Structure:")
    print(community_info.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    
    return analyzer, visualizer

# ==================== 测试 ====================
if __name__ == "__main__":
    print("Import this module and use run_full_analysis(graph_builder)")