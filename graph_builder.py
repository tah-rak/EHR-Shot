# """
# 1_graph_builder.py
# 构建EHRshot异构图

# 使用方法:
#     from graph_builder import build_ehrshot_graph
#     graph_builder = build_ehrshot_graph(data_path)
# """

# import os
# import pandas as pd
# import numpy as np
# import torch
# from torch_geometric.data import HeteroData
# from collections import defaultdict
# import warnings
# warnings.filterwarnings('ignore')

# # ==================== 数据加载器 ====================
# class EHRDataLoader:
#     def __init__(self, data_path):
#         self.data_path = data_path
        
#     def load_data(self):
#         """加载所有必要的数据表"""
#         print("Loading EHRshot data...")
        
#         # 核心临床事件表
#         self.person = pd.read_csv(os.path.join(self.data_path, "sampled_person.csv"))
#         self.condition = pd.read_csv(os.path.join(self.data_path, "sampled_condition_occurrence.csv"))
#         self.drug = pd.read_csv(os.path.join(self.data_path, "sampled_drug_exposure.csv"))
#         self.observation = pd.read_csv(os.path.join(self.data_path, "sampled_observation.csv"))
#         self.measurement = pd.read_csv(os.path.join(self.data_path, "sampled_measurement.csv"))
        
#         # 词汇表
#         self.concept = pd.read_csv(os.path.join(self.data_path, "concept.csv"))
        
#         # 预测标签
#         self.labels = pd.read_csv(os.path.join(self.data_path, "labeled_patients.csv"))
        
#         print(f"✓ Loaded {len(self.person)} patients")
#         print(f"✓ Loaded {len(self.condition)} condition records")
#         print(f"✓ Loaded {len(self.drug)} drug records")
#         print(f"✓ Loaded {len(self.observation)} observation records")
#         print(f"✓ Loaded {len(self.measurement)} measurement records")
        
#         return self

# # ==================== 异构图构建器 ====================
# class HeterogeneousGraphBuilder:
#     def __init__(self, data_loader):
#         self.dl = data_loader
        
#         # 节点映射字典
#         self.visit_map = {}
#         self.disease_map = {}
#         self.drug_map = {}
#         self.patient_map = {}
#         self.symptom_map = {}
        
#         # 概念名称映射
#         self.concept_names = dict(zip(self.dl.concept['concept_id'], 
#                                      self.dl.concept['concept_name']))
        
#     def extract_visits(self):
#         """从就诊事件中提取visit节点"""
#         print("\nExtracting visits...")
        
#         visits_from_condition = self.dl.condition[['visit_occurrence_id', 'condition_DATE', 
#                                                      'person_id']].dropna(subset=['visit_occurrence_id'])
#         visits_from_drug = self.dl.drug[['visit_occurrence_id', 'drug_exposure_start_DATE', 
#                                          'person_id']].dropna(subset=['visit_occurrence_id'])
        
#         all_visits = pd.concat([
#             visits_from_condition.rename(columns={'condition_DATE': 'visit_date'}),
#             visits_from_drug.rename(columns={'drug_exposure_start_DATE': 'visit_date'})
#         ])
        
#         all_visits = all_visits.groupby('visit_occurrence_id').agg({
#             'visit_date': 'first',
#             'person_id': 'first'
#         }).reset_index()
        
#         self.visit_map = {vid: idx for idx, vid in enumerate(all_visits['visit_occurrence_id'].unique())}
        
#         visit_features = []
#         for vid in all_visits['visit_occurrence_id'].unique():
#             visit_data = all_visits[all_visits['visit_occurrence_id'] == vid].iloc[0]
#             visit_features.append({
#                 'visit_id': vid,
#                 'person_id': visit_data['person_id'],
#                 'visit_date': visit_data['visit_date']
#             })
            
#         self.visit_df = pd.DataFrame(visit_features)
#         print(f"✓ Extracted {len(self.visit_map)} unique visits")
        
#         return self
    
#     def extract_diseases(self):
#         """提取疾病节点"""
#         print("\nExtracting diseases...")
        
#         disease_stats = self.dl.condition.groupby('condition_concept_id').agg({
#             'person_id': 'nunique',
#             'condition_occurrence_id': 'count'
#         }).reset_index()
        
#         disease_stats.columns = ['concept_id', 'num_patients', 'num_occurrences']
#         disease_stats['disease_name'] = disease_stats['concept_id'].map(self.concept_names)
#         disease_stats['disease_name'] = disease_stats['disease_name'].fillna('Unknown Disease')
        
#         self.disease_map = {cid: idx for idx, cid in enumerate(disease_stats['concept_id'])}
#         self.disease_df = disease_stats
        
#         print(f"✓ Extracted {len(self.disease_map)} unique diseases")
#         return self
    
#     def extract_drugs(self):
#         """提取药物节点，计算effectiveness指标"""
#         print("\nExtracting drugs...")
        
#         drug_stats = self.dl.drug.groupby('drug_concept_id').agg({
#             'person_id': 'nunique',
#             'visit_occurrence_id': 'nunique',
#             'drug_exposure_id': 'count'
#         }).reset_index()
        
#         drug_stats.columns = ['concept_id', 'num_patients', 'num_visits', 'num_prescriptions']
        
#         # 计算effectiveness: 如果ratio高说明同一患者需要多次就诊（可能效果不佳）
#         drug_stats['effectiveness_score'] = drug_stats['num_patients'] / drug_stats['num_visits']
        
#         drug_stats['drug_name'] = drug_stats['concept_id'].map(self.concept_names)
#         drug_stats['drug_name'] = drug_stats['drug_name'].fillna('Unknown Drug')
        
#         # 计算每个药物关联的疾病数
#         drug_disease_counts = self.dl.drug.merge(
#             self.dl.condition[['visit_occurrence_id', 'condition_concept_id']],
#             on='visit_occurrence_id', how='inner'
#         ).groupby('drug_concept_id')['condition_concept_id'].nunique().reset_index()
#         drug_disease_counts.columns = ['concept_id', 'num_diseases_treated']
        
#         drug_stats = drug_stats.merge(drug_disease_counts, on='concept_id', how='left')
#         drug_stats['num_diseases_treated'] = drug_stats['num_diseases_treated'].fillna(0)
        
#         self.drug_map = {cid: idx for idx, cid in enumerate(drug_stats['concept_id'])}
#         self.drug_df = drug_stats
        
#         print(f"✓ Extracted {len(self.drug_map)} unique drugs")
#         print(f"  Average diseases per drug: {drug_stats['num_diseases_treated'].mean():.2f}")
#         print(f"  Average effectiveness score: {drug_stats['effectiveness_score'].mean():.4f}")
        
#         return self
    
#     def extract_patients(self):
#         """提取患者节点"""
#         print("\nExtracting patients...")
        
#         patient_stats = self.dl.condition.groupby('person_id').agg({
#             'visit_occurrence_id': 'nunique',
#             'condition_occurrence_id': 'count'
#         }).reset_index()
#         patient_stats.columns = ['person_id', 'num_visits', 'num_conditions']
        
#         patient_stats = patient_stats.merge(
#             self.dl.person[['person_id', 'gender_concept_id', 'year_of_birth', 
#                            'race_concept_id', 'ethnicity_concept_id']],
#             on='person_id', how='left'
#         )
        
#         patient_stats['age'] = 2024 - patient_stats['year_of_birth']
        
#         patient_stats = patient_stats.merge(
#             self.dl.labels[['patient_id', 'value']].rename(columns={'patient_id': 'person_id'}),
#             on='person_id', how='left'
#         )
#         patient_stats['label'] = patient_stats['value'].map({True: 1, False: 0}).fillna(-1)
        
#         self.patient_map = {pid: idx for idx, pid in enumerate(patient_stats['person_id'])}
#         self.patient_df = patient_stats
        
#         print(f"✓ Extracted {len(self.patient_map)} unique patients")
#         print(f"  Average age: {patient_stats['age'].mean():.1f}")
#         print(f"  Labeled patients: {(patient_stats['label'] != -1).sum()}")
        
#         return self
    
#     def extract_symptoms(self):
#         """提取症状节点"""
#         print("\nExtracting symptoms...")
        
#         obs_stats = self.dl.observation.groupby('observation_concept_id').agg({
#             'person_id': 'nunique',
#             'observation_id': 'count'
#         }).reset_index()
#         obs_stats.columns = ['concept_id', 'num_patients', 'num_occurrences']
        
#         meas_stats = self.dl.measurement.groupby('measurement_concept_id').agg({
#             'person_id': 'nunique',
#             'measurement_id': 'count'
#         }).reset_index()
#         meas_stats.columns = ['concept_id', 'num_patients', 'num_occurrences']
        
#         symptom_stats = pd.concat([obs_stats, meas_stats])
#         symptom_stats = symptom_stats.groupby('concept_id').agg({
#             'num_patients': 'sum',
#             'num_occurrences': 'sum'
#         }).reset_index()
        
#         symptom_stats['symptom_name'] = symptom_stats['concept_id'].map(self.concept_names)
#         symptom_stats['symptom_name'] = symptom_stats['symptom_name'].fillna('Unknown Symptom')
        
#         self.symptom_map = {cid: idx for idx, cid in enumerate(symptom_stats['concept_id'])}
#         self.symptom_df = symptom_stats
        
#         print(f"✓ Extracted {len(self.symptom_map)} unique symptoms/measurements")
        
#         return self
    
#     def build_edges(self):
#         """构建所有类型的边"""
#         print("\nBuilding edges...")
        
#         edges = {
#             'visit_disease': [],
#             'visit_drug': [],
#             'visit_patient': [],
#             'visit_symptom': []
#         }
        
#         # 1. Visit -> Disease
#         condition_edges = self.dl.condition[['visit_occurrence_id', 'condition_concept_id']].dropna()
#         for _, row in condition_edges.iterrows():
#             if row['visit_occurrence_id'] in self.visit_map and row['condition_concept_id'] in self.disease_map:
#                 edges['visit_disease'].append([
#                     self.visit_map[row['visit_occurrence_id']],
#                     self.disease_map[row['condition_concept_id']]
#                 ])
        
#         # 2. Visit -> Drug
#         drug_edges = self.dl.drug[['visit_occurrence_id', 'drug_concept_id']].dropna()
#         for _, row in drug_edges.iterrows():
#             if row['visit_occurrence_id'] in self.visit_map and row['drug_concept_id'] in self.drug_map:
#                 edges['visit_drug'].append([
#                     self.visit_map[row['visit_occurrence_id']],
#                     self.drug_map[row['drug_concept_id']]
#                 ])
        
#         # 3. Visit -> Patient
#         for _, row in self.visit_df.iterrows():
#             if row['visit_id'] in self.visit_map and row['person_id'] in self.patient_map:
#                 edges['visit_patient'].append([
#                     self.visit_map[row['visit_id']],
#                     self.patient_map[row['person_id']]
#                 ])
        
#         # 4. Visit -> Symptom
#         obs_edges = self.dl.observation[['visit_occurrence_id', 'observation_concept_id']].dropna()
#         for _, row in obs_edges.iterrows():
#             if row['visit_occurrence_id'] in self.visit_map and row['observation_concept_id'] in self.symptom_map:
#                 edges['visit_symptom'].append([
#                     self.visit_map[row['visit_occurrence_id']],
#                     self.symptom_map[row['observation_concept_id']]
#                 ])
        
#         meas_edges = self.dl.measurement[['visit_occurrence_id', 'measurement_concept_id']].dropna()
#         for _, row in meas_edges.iterrows():
#             if row['visit_occurrence_id'] in self.visit_map and row['measurement_concept_id'] in self.symptom_map:
#                 edges['visit_symptom'].append([
#                     self.visit_map[row['visit_occurrence_id']],
#                     self.symptom_map[row['measurement_concept_id']]
#                 ])
        
#         # 转换为tensor
#         self.edge_index = {}
#         for edge_type, edge_list in edges.items():
#             if len(edge_list) > 0:
#                 self.edge_index[edge_type] = torch.tensor(edge_list, dtype=torch.long).t()
#                 print(f"✓ Built {len(edge_list)} {edge_type} edges")
#             else:
#                 self.edge_index[edge_type] = torch.tensor([[], []], dtype=torch.long)
        
#         return self
    
#     def build_pyg_hetero_graph(self):
#         """构建PyTorch Geometric异构图"""
#         print("\nBuilding PyG HeteroData...")
        
#         data = HeteroData()
        
#         # 添加节点
#         data['visit'].num_nodes = len(self.visit_map)
        
#         data['disease'].num_nodes = len(self.disease_map)
#         data['disease'].x = torch.tensor(
#             self.disease_df[['num_patients', 'num_occurrences']].values,
#             dtype=torch.float
#         )
        
#         data['drug'].num_nodes = len(self.drug_map)
#         data['drug'].x = torch.tensor(
#             self.drug_df[['num_patients', 'num_visits', 'num_prescriptions', 
#                          'effectiveness_score', 'num_diseases_treated']].values,
#             dtype=torch.float
#         )
        
#         data['patient'].num_nodes = len(self.patient_map)
#         data['patient'].x = torch.tensor(
#             self.patient_df[['num_visits', 'num_conditions', 'age', 
#                            'gender_concept_id', 'race_concept_id']].fillna(0).values,
#             dtype=torch.float
#         )
#         data['patient'].y = torch.tensor(self.patient_df['label'].values, dtype=torch.long)
        
#         data['symptom'].num_nodes = len(self.symptom_map)
#         data['symptom'].x = torch.tensor(
#             self.symptom_df[['num_patients', 'num_occurrences']].values,
#             dtype=torch.float
#         )
        
#         # 添加边
#         data['visit', 'diagnosed_with', 'disease'].edge_index = self.edge_index['visit_disease']
#         data['visit', 'prescribed', 'drug'].edge_index = self.edge_index['visit_drug']
#         data['visit', 'belongs_to', 'patient'].edge_index = self.edge_index['visit_patient']
#         data['visit', 'has_symptom', 'symptom'].edge_index = self.edge_index['visit_symptom']
        
#         # 反向边
#         data['disease', 'diagnosed_in', 'visit'].edge_index = data['visit', 'diagnosed_with', 'disease'].edge_index.flip(0)
#         data['drug', 'prescribed_in', 'visit'].edge_index = data['visit', 'prescribed', 'drug'].edge_index.flip(0)
#         data['patient', 'visited', 'visit'].edge_index = data['visit', 'belongs_to', 'patient'].edge_index.flip(0)
#         data['symptom', 'observed_in', 'visit'].edge_index = data['visit', 'has_symptom', 'symptom'].edge_index.flip(0)
        
#         self.hetero_data = data
        
#         print("\n" + "="*60)
#         print("Heterogeneous Graph Summary:")
#         print("="*60)
#         print(data)
#         print("="*60)
        
#         return self

# # ==================== 主函数 ====================
# def build_ehrshot_graph(data_path):
#     """主函数：构建完整的异构图"""
    
#     loader = EHRDataLoader(data_path).load_data()
    
#     builder = HeterogeneousGraphBuilder(loader)
#     builder.extract_visits()
#     builder.extract_diseases()
#     builder.extract_drugs()
#     builder.extract_patients()
#     builder.extract_symptoms()
#     builder.build_edges()
#     builder.build_pyg_hetero_graph()
    
#     return builder

# # ==================== 测试 ====================
# if __name__ == "__main__":
#     data_path = "/home/henry/Desktop/LLM/GraphML/data/"
#     graph_builder = build_ehrshot_graph(data_path)
#     print("\n✓ Graph construction complete!")
"""
graph_builder.py (修复版)
构建EHRshot异构图

🔧 修复内容：
1. condition_DATE → condition_start_DATE (第69行)
2. 添加了更详细的错误处理

使用方法:
    from graph_builder import build_ehrshot_graph
    graph_builder = build_ehrshot_graph(data_path)
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载器 ====================
class EHRDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load_data(self):
        """加载所有必要的数据表"""
        print("Loading EHRshot data...")
        
        try:
            # 核心临床事件表
            self.person = pd.read_csv(os.path.join(self.data_path, "sampled_person.csv"))
            self.condition = pd.read_csv(os.path.join(self.data_path, "sampled_condition_occurrence.csv"))
            self.drug = pd.read_csv(os.path.join(self.data_path, "sampled_drug_exposure.csv"))
            self.observation = pd.read_csv(os.path.join(self.data_path, "sampled_observation.csv"))
            self.measurement = pd.read_csv(os.path.join(self.data_path, "sampled_measurement.csv"))
            
            # 词汇表
            self.concept = pd.read_csv(os.path.join(self.data_path, "concept.csv"))
            
            # 预测标签
            self.labels = pd.read_csv(os.path.join(self.data_path, "labeled_patients.csv"))
            
            print(f"✓ Loaded {len(self.person)} patients")
            print(f"✓ Loaded {len(self.condition)} condition records")
            print(f"✓ Loaded {len(self.drug)} drug records")
            print(f"✓ Loaded {len(self.observation)} observation records")
            print(f"✓ Loaded {len(self.measurement)} measurement records")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Cannot find file - {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
        
        return self

# ==================== 异构图构建器 ====================
class HeterogeneousGraphBuilder:
    def __init__(self, data_loader):
        self.dl = data_loader
        
        # 节点映射字典
        self.visit_map = {}
        self.disease_map = {}
        self.drug_map = {}
        self.patient_map = {}
        self.symptom_map = {}
        
        # 概念名称映射
        self.concept_names = dict(zip(self.dl.concept['concept_id'], 
                                     self.dl.concept['concept_name']))
        
    def extract_visits(self):
        """从就诊事件中提取visit节点"""
        print("\nExtracting visits...")
        
        # 🔧 修改1: condition_DATE → condition_start_DATE
        visits_from_condition = self.dl.condition[['visit_occurrence_id', 'condition_start_DATE', 
                                                     'person_id']].dropna(subset=['visit_occurrence_id'])
        visits_from_drug = self.dl.drug[['visit_occurrence_id', 'drug_exposure_start_DATE', 
                                         'person_id']].dropna(subset=['visit_occurrence_id'])
        
        all_visits = pd.concat([
            visits_from_condition.rename(columns={'condition_start_DATE': 'visit_date'}),
            visits_from_drug.rename(columns={'drug_exposure_start_DATE': 'visit_date'})
        ])
        
        all_visits = all_visits.groupby('visit_occurrence_id').agg({
            'visit_date': 'first',
            'person_id': 'first'
        }).reset_index()
        
        self.visit_map = {vid: idx for idx, vid in enumerate(all_visits['visit_occurrence_id'].unique())}
        
        visit_features = []
        for vid in all_visits['visit_occurrence_id'].unique():
            visit_data = all_visits[all_visits['visit_occurrence_id'] == vid].iloc[0]
            visit_features.append({
                'visit_id': vid,
                'person_id': visit_data['person_id'],
                'visit_date': visit_data['visit_date']
            })
            
        self.visit_df = pd.DataFrame(visit_features)
        print(f"✓ Extracted {len(self.visit_map)} unique visits")
        
        return self
    
    def extract_diseases(self):
        """提取疾病节点"""
        print("\nExtracting diseases...")
        
        disease_stats = self.dl.condition.groupby('condition_concept_id').agg({
            'person_id': 'nunique',
            'condition_occurrence_id': 'count'
        }).reset_index()
        
        disease_stats.columns = ['concept_id', 'num_patients', 'num_occurrences']
        disease_stats['disease_name'] = disease_stats['concept_id'].map(self.concept_names)
        disease_stats['disease_name'] = disease_stats['disease_name'].fillna('Unknown Disease')
        
        self.disease_map = {cid: idx for idx, cid in enumerate(disease_stats['concept_id'])}
        self.disease_df = disease_stats
        
        print(f"✓ Extracted {len(self.disease_map)} unique diseases")
        return self
    
    def extract_drugs(self):
        """提取药物节点，计算effectiveness指标"""
        print("\nExtracting drugs...")
        
        drug_stats = self.dl.drug.groupby('drug_concept_id').agg({
            'person_id': 'nunique',
            'visit_occurrence_id': 'nunique',
            'drug_exposure_id': 'count'
        }).reset_index()
        
        drug_stats.columns = ['concept_id', 'num_patients', 'num_visits', 'num_prescriptions']
        
        # 计算effectiveness: 如果ratio高说明同一患者需要多次就诊（可能效果不佳）
        drug_stats['effectiveness_score'] = drug_stats['num_patients'] / drug_stats['num_visits']
        
        drug_stats['drug_name'] = drug_stats['concept_id'].map(self.concept_names)
        drug_stats['drug_name'] = drug_stats['drug_name'].fillna('Unknown Drug')
        
        # 计算每个药物关联的疾病数
        drug_disease_counts = self.dl.drug.merge(
            self.dl.condition[['visit_occurrence_id', 'condition_concept_id']],
            on='visit_occurrence_id', how='inner'
        ).groupby('drug_concept_id')['condition_concept_id'].nunique().reset_index()
        drug_disease_counts.columns = ['concept_id', 'num_diseases_treated']
        
        drug_stats = drug_stats.merge(drug_disease_counts, on='concept_id', how='left')
        drug_stats['num_diseases_treated'] = drug_stats['num_diseases_treated'].fillna(0)
        
        self.drug_map = {cid: idx for idx, cid in enumerate(drug_stats['concept_id'])}
        self.drug_df = drug_stats
        
        print(f"✓ Extracted {len(self.drug_map)} unique drugs")
        print(f"  Average diseases per drug: {drug_stats['num_diseases_treated'].mean():.2f}")
        print(f"  Average effectiveness score: {drug_stats['effectiveness_score'].mean():.4f}")
        
        return self
    
    def extract_patients(self):
        """提取患者节点"""
        print("\nExtracting patients...")
        
        patient_stats = self.dl.condition.groupby('person_id').agg({
            'visit_occurrence_id': 'nunique',
            'condition_occurrence_id': 'count'
        }).reset_index()
        patient_stats.columns = ['person_id', 'num_visits', 'num_conditions']
        
        patient_stats = patient_stats.merge(
            self.dl.person[['person_id', 'gender_concept_id', 'year_of_birth', 
                           'race_concept_id', 'ethnicity_concept_id']],
            on='person_id', how='left'
        )
        
        patient_stats['age'] = 2024 - patient_stats['year_of_birth']
        
        patient_stats = patient_stats.merge(
            self.dl.labels[['patient_id', 'value']].rename(columns={'patient_id': 'person_id'}),
            on='person_id', how='left'
        )
        patient_stats['label'] = patient_stats['value'].map({True: 1, False: 0}).fillna(-1)
        
        self.patient_map = {pid: idx for idx, pid in enumerate(patient_stats['person_id'])}
        self.patient_df = patient_stats
        
        print(f"✓ Extracted {len(self.patient_map)} unique patients")
        print(f"  Average age: {patient_stats['age'].mean():.1f}")
        print(f"  Labeled patients: {(patient_stats['label'] != -1).sum()}")
        
        return self
    
    def extract_symptoms(self):
        """提取症状节点"""
        print("\nExtracting symptoms...")
        
        obs_stats = self.dl.observation.groupby('observation_concept_id').agg({
            'person_id': 'nunique',
            'observation_id': 'count'
        }).reset_index()
        obs_stats.columns = ['concept_id', 'num_patients', 'num_occurrences']
        
        meas_stats = self.dl.measurement.groupby('measurement_concept_id').agg({
            'person_id': 'nunique',
            'measurement_id': 'count'
        }).reset_index()
        meas_stats.columns = ['concept_id', 'num_patients', 'num_occurrences']
        
        symptom_stats = pd.concat([obs_stats, meas_stats])
        symptom_stats = symptom_stats.groupby('concept_id').agg({
            'num_patients': 'sum',
            'num_occurrences': 'sum'
        }).reset_index()
        
        symptom_stats['symptom_name'] = symptom_stats['concept_id'].map(self.concept_names)
        symptom_stats['symptom_name'] = symptom_stats['symptom_name'].fillna('Unknown Symptom')
        
        self.symptom_map = {cid: idx for idx, cid in enumerate(symptom_stats['concept_id'])}
        self.symptom_df = symptom_stats
        
        print(f"✓ Extracted {len(self.symptom_map)} unique symptoms/measurements")
        
        return self
    
    def build_edges(self):
        """构建所有类型的边"""
        print("\nBuilding edges...")
        
        edges = {
            'visit_disease': [],
            'visit_drug': [],
            'visit_patient': [],
            'visit_symptom': []
        }
        
        # 1. Visit -> Disease
        print("  Building visit-disease edges...")
        condition_edges = self.dl.condition[['visit_occurrence_id', 'condition_concept_id']].dropna()
        for _, row in condition_edges.iterrows():
            if row['visit_occurrence_id'] in self.visit_map and row['condition_concept_id'] in self.disease_map:
                edges['visit_disease'].append([
                    self.visit_map[row['visit_occurrence_id']],
                    self.disease_map[row['condition_concept_id']]
                ])
        
        # 2. Visit -> Drug
        print("  Building visit-drug edges...")
        drug_edges = self.dl.drug[['visit_occurrence_id', 'drug_concept_id']].dropna()
        for _, row in drug_edges.iterrows():
            if row['visit_occurrence_id'] in self.visit_map and row['drug_concept_id'] in self.drug_map:
                edges['visit_drug'].append([
                    self.visit_map[row['visit_occurrence_id']],
                    self.drug_map[row['drug_concept_id']]
                ])
        
        # 3. Visit -> Patient
        print("  Building visit-patient edges...")
        for _, row in self.visit_df.iterrows():
            if row['visit_id'] in self.visit_map and row['person_id'] in self.patient_map:
                edges['visit_patient'].append([
                    self.visit_map[row['visit_id']],
                    self.patient_map[row['person_id']]
                ])
        
        # 4. Visit -> Symptom
        print("  Building visit-symptom edges...")
        obs_edges = self.dl.observation[['visit_occurrence_id', 'observation_concept_id']].dropna()
        for _, row in obs_edges.iterrows():
            if row['visit_occurrence_id'] in self.visit_map and row['observation_concept_id'] in self.symptom_map:
                edges['visit_symptom'].append([
                    self.visit_map[row['visit_occurrence_id']],
                    self.symptom_map[row['observation_concept_id']]
                ])
        
        meas_edges = self.dl.measurement[['visit_occurrence_id', 'measurement_concept_id']].dropna()
        for _, row in meas_edges.iterrows():
            if row['visit_occurrence_id'] in self.visit_map and row['measurement_concept_id'] in self.symptom_map:
                edges['visit_symptom'].append([
                    self.visit_map[row['visit_occurrence_id']],
                    self.symptom_map[row['measurement_concept_id']]
                ])
        
        # 转换为tensor
        self.edge_index = {}
        for edge_type, edge_list in edges.items():
            if len(edge_list) > 0:
                self.edge_index[edge_type] = torch.tensor(edge_list, dtype=torch.long).t()
                print(f"✓ Built {len(edge_list)} {edge_type} edges")
            else:
                self.edge_index[edge_type] = torch.tensor([[], []], dtype=torch.long)
                print(f"⚠ Warning: No {edge_type} edges found")
        
        return self
    
    def build_pyg_hetero_graph(self):
        """构建PyTorch Geometric异构图"""
        print("\nBuilding PyG HeteroData...")
        
        data = HeteroData()
        
        # 添加节点
        data['visit'].num_nodes = len(self.visit_map)
        
        data['disease'].num_nodes = len(self.disease_map)
        data['disease'].x = torch.tensor(
            self.disease_df[['num_patients', 'num_occurrences']].values,
            dtype=torch.float
        )
        
        data['drug'].num_nodes = len(self.drug_map)
        data['drug'].x = torch.tensor(
            self.drug_df[['num_patients', 'num_visits', 'num_prescriptions', 
                         'effectiveness_score', 'num_diseases_treated']].values,
            dtype=torch.float
        )
        
        data['patient'].num_nodes = len(self.patient_map)
        data['patient'].x = torch.tensor(
            self.patient_df[['num_visits', 'num_conditions', 'age', 
                           'gender_concept_id', 'race_concept_id']].fillna(0).values,
            dtype=torch.float
        )
        data['patient'].y = torch.tensor(self.patient_df['label'].values, dtype=torch.long)
        
        data['symptom'].num_nodes = len(self.symptom_map)
        data['symptom'].x = torch.tensor(
            self.symptom_df[['num_patients', 'num_occurrences']].values,
            dtype=torch.float
        )
        
        # 添加边
        data['visit', 'diagnosed_with', 'disease'].edge_index = self.edge_index['visit_disease']
        data['visit', 'prescribed', 'drug'].edge_index = self.edge_index['visit_drug']
        data['visit', 'belongs_to', 'patient'].edge_index = self.edge_index['visit_patient']
        data['visit', 'has_symptom', 'symptom'].edge_index = self.edge_index['visit_symptom']
        
        # 反向边
        data['disease', 'diagnosed_in', 'visit'].edge_index = data['visit', 'diagnosed_with', 'disease'].edge_index.flip(0)
        data['drug', 'prescribed_in', 'visit'].edge_index = data['visit', 'prescribed', 'drug'].edge_index.flip(0)
        data['patient', 'visited', 'visit'].edge_index = data['visit', 'belongs_to', 'patient'].edge_index.flip(0)
        data['symptom', 'observed_in', 'visit'].edge_index = data['visit', 'has_symptom', 'symptom'].edge_index.flip(0)
        
        self.hetero_data = data
        
        print("\n" + "="*60)
        print("Heterogeneous Graph Summary:")
        print("="*60)
        print(data)
        print("="*60)
        
        return self

# ==================== 主函数 ====================
def build_ehrshot_graph(data_path):
    """主函数：构建完整的异构图"""
    
    loader = EHRDataLoader(data_path).load_data()
    
    builder = HeterogeneousGraphBuilder(loader)
    builder.extract_visits()
    builder.extract_diseases()
    builder.extract_drugs()
    builder.extract_patients()
    builder.extract_symptoms()
    builder.build_edges()
    builder.build_pyg_hetero_graph()
    
    return builder

# ==================== 测试 ====================
if __name__ == "__main__":
    data_path = "/home/henry/Desktop/LLM/GraphML/data/"
    graph_builder = build_ehrshot_graph(data_path)
    print("\n✓ Graph construction complete!")