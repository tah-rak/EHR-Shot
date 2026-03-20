"""
3_run_analysis.py
主执行文件 - 运行完整的EHRshot网络分析

直接运行: python 3_run_analysis.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 导入自定义模块
from graph_builder import build_ehrshot_graph
from network_analyzer import run_full_analysis

# ==================== 配置 ====================
# 修改为你的实际数据路径
DATA_PATH = "/home/henry/Desktop/LLM/GraphML/data/"
OUTPUT_DIR = "./analysis_results"

# ==================== 主执行函数 ====================
def main():
    """主执行函数"""
    
    print("="*80)
    print("EHRshot Network Analysis - Project 1")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'graphs'), exist_ok=True)
    
    # Step 1: 构建图
    print("\n[STEP 1/3] Building heterogeneous graph...")
    print("-" * 80)
    graph_builder = build_ehrshot_graph(DATA_PATH)
    
    # Step 2: 网络分析
    print("\n[STEP 2/3] Running network analysis...")
    print("-" * 80)
    analyzer, visualizer = run_full_analysis(graph_builder)
    
    # Step 3: 导出结果
    print("\n[STEP 3/3] Exporting results...")
    print("-" * 80)
    export_results(graph_builder, analyzer, visualizer)
    
    # 完成
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"\n📁 Results saved in: {OUTPUT_DIR}/")
    print("  ├── tables/          (CSV files)")
    print("  ├── figures/         (PNG images)")
    print("  └── graphs/          (NetworkX files)")
    print("\n" + "="*80)
    
    return graph_builder, analyzer, visualizer

# ==================== 导出函数 ====================
def export_results(graph_builder, analyzer, visualizer):
    """导出所有结果"""
    
    # 1. 导出表格
    print("\n  Exporting tables...")
    tables_dir = os.path.join(OUTPUT_DIR, 'tables')
    
    analyzer.drug_metrics_df.to_csv(
        os.path.join(tables_dir, 'drug_effectiveness_metrics.csv'), index=False
    )
    print(f"    ✓ drug_effectiveness_metrics.csv")
    
    analyzer.community_df.to_csv(
        os.path.join(tables_dir, 'community_structure.csv'), index=False
    )
    print(f"    ✓ community_structure.csv")
    
    graph_builder.disease_df.to_csv(
        os.path.join(tables_dir, 'disease_statistics.csv'), index=False
    )
    print(f"    ✓ disease_statistics.csv")
    
    graph_builder.patient_df.to_csv(
        os.path.join(tables_dir, 'patient_statistics.csv'), index=False
    )
    print(f"    ✓ patient_statistics.csv")
    
    # 2. 保存可视化
    print("\n  Saving visualizations...")
    figures_dir = os.path.join(OUTPUT_DIR, 'figures')
    
    # 药物有效性
    fig1 = visualizer.plot_drug_effectiveness_analysis(top_n=20)
    fig1.savefig(os.path.join(figures_dir, 'fig1_drug_effectiveness.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"    ✓ fig1_drug_effectiveness.png")
    
    # 网络可视化
    fig2 = visualizer.plot_network_sample(sample_size=100)
    fig2.savefig(os.path.join(figures_dir, 'fig2_network_visualization.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"    ✓ fig2_network_visualization.png")
    
    # 社区结构
    fig3 = visualizer.plot_community_structure()
    fig3.savefig(os.path.join(figures_dir, 'fig3_community_structure.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"    ✓ fig3_community_structure.png")
    
    # 3. 导出图文件
    print("\n  Exporting graph files...")
    graphs_dir = os.path.join(OUTPUT_DIR, 'graphs')
    
    import networkx as nx
    G = analyzer.nx_graph
    
    # 节点列表
    nodes_data = []
    for node in G.nodes():
        node_data = {'node_id': node}
        node_data.update(G.nodes[node])
        nodes_data.append(node_data)
    
    nodes_df = pd.DataFrame(nodes_data)
    nodes_df.to_csv(os.path.join(graphs_dir, 'nodes.csv'), index=False)
    print(f"    ✓ nodes.csv ({len(nodes_df)} nodes)")
    
    # 边列表
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'source': u,
            'target': v,
            'weight': data['weight']
        })
    
    edges_df = pd.DataFrame(edges_data)
    edges_df.to_csv(os.path.join(graphs_dir, 'edges.csv'), index=False)
    print(f"    ✓ edges.csv ({len(edges_df)} edges)")
    
    # GraphML格式
    nx.write_graphml(G, os.path.join(graphs_dir, 'drug_disease_network.graphml'))
    print(f"    ✓ drug_disease_network.graphml")
    
    # 4. 生成简要报告
    print("\n  Generating summary report...")
    generate_summary_report(graph_builder, analyzer)
    print(f"    ✓ SUMMARY_REPORT.txt")

def generate_summary_report(graph_builder, analyzer):
    """生成简要文本报告"""
    
    report_path = os.path.join(OUTPUT_DIR, 'SUMMARY_REPORT.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EHRshot Network Analysis - Summary Report\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("RESEARCH QUESTION:\n")
        f.write("What is the relationship between a drug's usage across multiple\n")
        f.write("diseases and its effectiveness in treating them?\n\n")
        
        f.write("="*80 + "\n")
        f.write("NETWORK STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Total Visits:    {len(graph_builder.visit_map):,}\n")
        f.write(f"Total Diseases:  {len(graph_builder.disease_map):,}\n")
        f.write(f"Total Drugs:     {len(graph_builder.drug_map):,}\n")
        f.write(f"Total Patients:  {len(graph_builder.patient_map):,}\n")
        f.write(f"Total Symptoms:  {len(graph_builder.symptom_map):,}\n\n")
        
        G = analyzer.nx_graph
        f.write(f"Network Nodes:   {G.number_of_nodes():,}\n")
        f.write(f"Network Edges:   {G.number_of_edges():,}\n")
        f.write(f"Communities:     {len(analyzer.communities)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("TOP 10 MOST EFFECTIVE DRUGS\n")
        f.write("="*80 + "\n")
        top_drugs = analyzer.drug_metrics_df.nlargest(10, 'composite_score')
        for i, (_, row) in enumerate(top_drugs.iterrows(), 1):
            f.write(f"{i:2d}. {row['drug_name'][:50]}\n")
            f.write(f"    Diseases: {row['num_diseases_connected']}, ")
            f.write(f"Effectiveness: {row['effectiveness_score']:.4f}, ")
            f.write(f"Score: {row['composite_score']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n")
        
        df = analyzer.drug_metrics_df
        f.write(f"• Average diseases per drug: {df['num_diseases_connected'].mean():.2f}\n")
        f.write(f"• Average effectiveness score: {df['effectiveness_score'].mean():.4f}\n")
        f.write(f"• Drugs treating >10 diseases: {(df['num_diseases_connected'] > 10).sum()}\n")
        
        versatile = df[df['num_diseases_connected'] > df['num_diseases_connected'].quantile(0.75)]
        low_eff = versatile[versatile['effectiveness_score'] < versatile['effectiveness_score'].median()]
        f.write(f"• Versatile but less effective drugs: {len(low_eff)}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FILES GENERATED\n")
        f.write("="*80 + "\n")
        f.write("tables/\n")
        f.write("  ├── drug_effectiveness_metrics.csv\n")
        f.write("  ├── community_structure.csv\n")
        f.write("  ├── disease_statistics.csv\n")
        f.write("  └── patient_statistics.csv\n\n")
        f.write("figures/\n")
        f.write("  ├── fig1_drug_effectiveness.png\n")
        f.write("  ├── fig2_network_visualization.png\n")
        f.write("  └── fig3_community_structure.png\n\n")
        f.write("graphs/\n")
        f.write("  ├── nodes.csv\n")
        f.write("  ├── edges.csv\n")
        f.write("  └── drug_disease_network.graphml\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

# ==================== 运行 ====================
if __name__ == "__main__":
    try:
        graph_builder, analyzer, visualizer = main()
        
        # 显示快速统计
        print("\n📊 Quick Statistics:")
        print(f"  • Network Nodes: {analyzer.nx_graph.number_of_nodes():,}")
        print(f"  • Network Edges: {analyzer.nx_graph.number_of_edges():,}")
        print(f"  • Communities: {len(analyzer.communities)}")
        
        top_drug = analyzer.drug_metrics_df.nlargest(1, 'composite_score').iloc[0]
        print(f"\n🏆 Top Drug: {top_drug['drug_name']}")
        print(f"  • Treats {top_drug['num_diseases_connected']} diseases")
        print(f"  • Effectiveness: {top_drug['effectiveness_score']:.4f}")
        print(f"  • Composite Score: {top_drug['composite_score']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)