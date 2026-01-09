#!/usr/bin/env python3
"""
lncRNA生物标志物发现流程 - 使用真实TCGA-LIHC数据

使用真实的TCGA数据识别区分肝癌肿瘤和正常样本的lncRNA生物标志物

数据源:
1. TCGA-LIHC.star_counts.tsv.gz - 基因表达矩阵 (log2转换后的counts)
2. TCGA-LIHC.clinical.tsv.gz - 临床/表型数据
3. gencode.v36.long_noncoding_RNAs.gtf.gz - lncRNA注释

作者: Bioinformatics Ph.D. Student
日期: 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import gzip
import re
from pathlib import Path

# 导入功能模块
from modules.step1_preprocessing import TCGAPreprocessor
from modules.step2_annotation_filter import LncRNAAnnotator
from modules.step3_differential_expression import DifferentialExpressionAnalyzer
from modules.step4_feature_selection import FeatureSelector
from modules.step5_classification import BiomarkerClassifier
from modules.step6_visualization import BiomarkerVisualizer

import config


def parse_gtf_for_lncrnas(gtf_file: str) -> set:
    """
    解析GTF文件,提取lncRNA基因ID

    Parameters:
    -----------
    gtf_file : str
        GTF文件路径(.gz压缩)

    Returns:
    --------
    set : lncRNA基因ID集合(无版本号)
    """
    print("\n解析GTF文件,提取lncRNA注释...")
    lncrna_genes = set()

    with gzip.open(gtf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            # 仅处理gene行
            if fields[2] != 'gene':
                continue

            # 解析属性字段
            attributes = fields[8]
            gene_type_match = re.search(r'gene_type "([^"]+)"', attributes)
            gene_id_match = re.search(r'gene_id "ENSG\d+\.?\d*"', attributes)

            if gene_type_match and gene_id_match:
                gene_type = gene_type_match.group(1)
                gene_id_full = gene_id_match.group(0)

                # 仅保留lncRNA
                if gene_type == 'lncRNA':
                    # 提取基因ID并去除版本号
                    gene_id = re.search(r'ENSG\d+', gene_id_full).group(0)
                    lncrna_genes.add(gene_id)

    print(f"✓ 从GTF中提取 {len(lncrna_genes):,} 个lncRNA基因")
    return lncrna_genes


def load_expression_matrix(expr_file: str, lncrna_genes: set) -> pd.DataFrame:
    """
    加载并处理表达矩阵

    Parameters:
    -----------
    expr_file : str
        表达矩阵文件路径(.tsv.gz)
    lncrna_genes : set
        lncRNA基因ID集合

    Returns:
    --------
    pd.DataFrame : 仅包含lncRNA的表达矩阵
    """
    print("\n加载基因表达矩阵...")

    # 读取TSV文件
    print("  - 读取文件(可能需要1-2分钟)...")
    expr_df = pd.read_csv(expr_file, sep='\t', compression='gzip', index_col=0)

    print(f"  - 原始维度: {expr_df.shape[0]:,} 基因 × {expr_df.shape[1]:,} 样本")

    # 去除Ensembl ID的版本号
    print("  - 去除Ensembl ID版本号...")
    expr_df.index = expr_df.index.str.split('.').str[0]

    # 过滤仅保留lncRNA
    print(f"  - 过滤lncRNA...")
    lncrna_present = [gid for gid in lncrna_genes if gid in expr_df.index]
    expr_lncrna = expr_df.loc[lncrna_present]

    print(f"✓ 表达矩阵加载完成: {expr_lncrna.shape[0]:,} lncRNA × {expr_lncrna.shape[1]:,} 样本")

    return expr_lncrna


def parse_sample_clinical_data(clinical_file: str) -> pd.DataFrame:
    """
    解析临床数据,获取样本类型

    Parameters:
    -----------
    clinical_file : str
        临床数据文件路径(.tsv.gz)

    Returns:
    --------
    pd.DataFrame : 样本类型信息
    """
    print("\n解析临床数据...")

    # 读取临床数据
    clinical_df = pd.read_csv(clinical_file, sep='\t', compression='gzip')

    # 第一列是sample ID (TCGA barcode格式)
    sample_id_col = clinical_df.columns[0]  # 'sample'

    # 从TCGA barcode解析样本类型
    print(f"  - 从TCGA barcode解析样本类型...")

    tumor_samples = []
    normal_samples = []

    for sid in clinical_df[sample_id_col].values:
        if not isinstance(sid, str) or len(sid) < 4:
            continue

        # TCGA barcode: 第14-15位是样本类型 (如01A, 11A)
        parts = sid.split('-')
        if len(parts) >= 4:
            sample_code = parts[3][:2]

            if sample_code in ['01', '02', '03', '04', '05', '06', '07', '08', '09']:
                tumor_samples.append(sid)
            elif sample_code in ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19']:
                normal_samples.append(sid)

    sample_info = pd.DataFrame({
        'sample_id': tumor_samples + normal_samples,
        'sample_type': ['Tumor'] * len(tumor_samples) + ['Normal'] * len(normal_samples)
    })

    tumor_count = len(tumor_samples)
    normal_count = len(normal_samples)

    print(f"✓ 临床数据解析完成:")
    print(f"  - 总样本数: {len(sample_info)}")
    print(f"  - 肿瘤: {tumor_count}")
    print(f"  - 正常: {normal_count}")

    return sample_info


def main():
    """主执行流程"""

    print("\n" + "=" * 70)
    print(" "*15 + "lncRNA生物标志物发现流程")
    print(" "*10 + "基于真实TCGA-LIHC数据分析")
    print("=" * 70)

    # ========================================================================
    # 步骤1: 加载数据
    # ========================================================================

    print("\n" + "=" * 70)
    print("步骤1: 数据加载")
    print("=" * 70)

    # 定义数据文件路径
    base_dir = Path(config.PROJECT_DIR)
    expr_file = base_dir / "TCGA-LIHC.star_counts.tsv.gz"
    clinical_file = base_dir / "TCGA-LIHC.clinical.tsv.gz"
    gtf_file = base_dir / "gencode.v36.long_noncoding_RNAs.gtf.gz"

    # 检查文件是否存在
    for f in [expr_file, clinical_file, gtf_file]:
        if not f.exists():
            print(f"\n错误: 文件不存在 - {f}")
            print("\n请确保以下文件在项目目录中:")
            print("  - TCGA-LIHC.star_counts.tsv.gz")
            print("  - TCGA-LIHC.clinical.tsv.gz")
            print("  - gencode.v36.long_noncoding_RNAs.gtf.gz")
            sys.exit(1)

    # 1.1 解析GTF获取lncRNA列表
    lncrna_genes = parse_gtf_for_lncrnas(str(gtf_file))

    # 1.2 加载表达矩阵
    expr_data = load_expression_matrix(str(expr_file), lncrna_genes)

    # 1.3 解析临床数据
    sample_info = parse_sample_clinical_data(str(clinical_file))

    # ========================================================================
    # 步骤2: 数据预处理和样本分类
    # ========================================================================

    print("\n" + "=" * 70)
    print("步骤2: 样本分类和数据预处理")
    print("=" * 70)

    # 匹配样本ID
    common_samples = [s for s in expr_data.columns if s in sample_info['sample_id'].values]
    print(f"\n匹配的样本数: {len(common_samples)}")

    if len(common_samples) == 0:
        print("\n错误: 表达数据和临床数据的样本ID不匹配!")
        print(f"\n表达数据样本示例: {expr_data.columns[0]}")
        print(f"临床数据样本示例: {sample_info['sample_id'].iloc[0]}")
        sys.exit(1)

    # 过滤数据
    expr_data = expr_data[common_samples]
    sample_info = sample_info[sample_info['sample_id'].isin(common_samples)].copy()

    # 创建样本类型映射
    sample_type_map = dict(zip(sample_info['sample_id'], sample_info['sample_type']))

    # 分离肿瘤和正常样本
    tumor_samples = sample_info[sample_info['sample_type'] == 'Tumor']['sample_id'].tolist()
    normal_samples = sample_info[sample_info['sample_type'] == 'Normal']['sample_id'].tolist()

    print(f"\n✓ 样本分类完成:")
    print(f"  - 肿瘤样本: {len(tumor_samples)}")
    print(f"  - 正常样本: {len(normal_samples)}")

    # 分离表达数据
    tumor_data = expr_data[tumor_samples]
    normal_data = expr_data[normal_samples]

    # 过滤低表达lncRNA
    print("\n过滤低表达lncRNA...")
    mean_expr = expr_data.mean(axis=1)
    expr_data_filtered = expr_data[mean_expr >= 1.0]
    tumor_data = tumor_data.loc[expr_data_filtered.index]
    normal_data = normal_data.loc[expr_data_filtered.index]

    print(f"✓ 保留 {expr_data_filtered.shape[0]:,} 个表达的lncRNA")

    # ========================================================================
    # 步骤3: 差异表达分析
    # ========================================================================

    print("\n" + "=" * 70)
    print("步骤3: 差异表达分析")
    print("=" * 70)

    de_analyzer = DifferentialExpressionAnalyzer(verbose=True)
    de_results = de_analyzer.identify_differential_expression(
        tumor_data, normal_data,
        log2fc_threshold=1.0,
        fdr_threshold=0.05,
        method='wilcoxon'
    )

    # 保存结果
    de_analyzer.save_results()

    significant_lncrnas = de_results[de_results['significant']]['gene_id'].tolist()
    print(f"\n✓ 发现 {len(significant_lncrnas)} 个显著差异表达lncRNA")

    # ========================================================================
    # 步骤4: 特征选择
    # ========================================================================

    print("\n" + "=" * 70)
    print("步骤4: 特征选择")
    print("=" * 70)

    feature_selector = FeatureSelector(verbose=True)
    selected_biomarkers, biomarker_matrix = feature_selector.run_feature_selection_pipeline(
        tumor_data, normal_data,
        de_genes=significant_lncrnas,
        n_features=20,
        method='lasso'
    )

    print(f"\n✓ 选择 {len(selected_biomarkers)} 个lncRNA生物标志物")
    for i, biomarker in enumerate(selected_biomarkers, 1):
        print(f"  {i}. {biomarker}")

    # ========================================================================
    # 步骤5: 机器学习分类
    # ========================================================================

    print("\n" + "=" * 70)
    print("步骤5: 机器学习分类 (SVM)")
    print("=" * 70)

    # 使用选择的生物标志物
    biomarker_tumor = biomarker_matrix[tumor_samples]
    biomarker_normal = biomarker_matrix[normal_samples]

    classifier = BiomarkerClassifier(classifier_type='svm', verbose=True)
    eval_results = classifier.run_classification_pipeline(biomarker_tumor, biomarker_normal)

    # ========================================================================
    # 步骤6: 可视化
    # ========================================================================

    print("\n" + "=" * 70)
    print("步骤6: 结果可视化")
    print("=" * 70)

    visualizer = BiomarkerVisualizer(verbose=True)

    # 火山图
    visualizer.plot_volcano(de_results)

    # 热图
    visualizer.plot_heatmap(
        biomarker_matrix,
        tumor_samples,
        normal_samples,
        n_biomarkers=20
    )

    # ROC曲线
    visualizer.plot_roc_curve(eval_results)

    # 综合总结图
    visualizer.create_summary_figure(
        de_results,
        biomarker_matrix,
        tumor_samples,
        normal_samples,
        eval_results
    )

    # ========================================================================
    # 总结报告
    # ========================================================================

    print("\n" + "=" * 70)
    print("流程完成! 结果总结")
    print("=" * 70)

    print(f"\n📊 分析统计:")
    print(f"  - 总lncRNA数: {expr_data_filtered.shape[0]:,}")
    print(f"  - 差异表达lncRNA: {len(significant_lncrnas)}")
    print(f"  - 选择标志物: {len(selected_biomarkers)}")

    print(f"\n🎯 分类器性能 (SVM):")
    print(f"  - 准确率: {eval_results['accuracy']:.3f}")
    print(f"  - 灵敏度: {eval_results['sensitivity']:.3f}")
    print(f"  - 特异度: {eval_results['specificity']:.3f}")
    if eval_results.get('auc'):
        print(f"  - AUC: {eval_results['auc']:.3f}")

    print(f"\n📁 输出文件:")
    print(f"  - 结果目录: {config.RESULTS_DIR}/")
    print(f"  - 图表目录: {config.FIGURES_DIR}/")

    print(f"\n✅ 成功完成lncRNA生物标志物发现流程!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
