#!/usr/bin/env python3
"""
主执行脚本 - lncRNA生物标志物发现流程

完整的端到端机器学习项目,用于识别区分TCGA-LIHC肿瘤与正常样本的lncRNA标志物

流程:
  Step 1: 数据预处理 (加载,清洗,转换)
  Step 2: lncRNA注释过滤
  Step 3: 差异表达分析
  Step 4: 特征选择 (Lasso/RF)
  Step 5: 机器学习分类 (SVM/RF)
  Step 6: 结果可视化

使用方法:
  python main_pipeline.py --data_type synthetic
  python main_pipeline.py --data_type real --expr data/expression.csv --phenotype data/phenotype.csv

作者: Bioinformatics Ph.D. Student
日期: 2025
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 导入config
import config

# 导入各功能模块
sys.path.insert(0, config.MODULES_DIR)
from modules.step1_preprocessing import TCGAPreprocessor
from modules.step2_annotation_filter import LncRNAAnnotator, generate_synthetic_lncRNA_ids
from modules.step3_differential_expression import DifferentialExpressionAnalyzer
from modules.step4_feature_selection import FeatureSelector
from modules.step5_classification import BiomarkerClassifier
from modules.step6_visualization import BiomarkerVisualizer


def generate_synthetic_tcga_data():
    """
    生成模拟的TCGA-LIHC数据

    用于演示和测试,数据特征:
    - 150个肿瘤样本, 50个正常样本
    - 15,000个基因 (其中5,000个lncRNA)
    - 约500个差异表达基因
    - 表达值近似log2正态分布

    Returns:
    --------
    expr_file : str
        表达矩阵文件路径
    pheno_file : str
        样本注释文件路径
    """
    print("\n" + "=" * 60)
    print("生成模拟TCGA-LIHC数据")
    print("=" * 60)

    # 设置随机种子
    np.random.seed(config.RANDOM_STATE)

    # 生成lncRNA ID
    lncrna_ids = generate_synthetic_lncRNA_ids(n_genes=config.SYNTHETIC_LNCRNA_N)

    # 生成mRNA ID
    mrna_ids = [f'ENSG{i:011d}' for i in range(1, config.SYNTHETIC_MRNA_ID_END)]

    # 合并所有基因ID
    all_gene_ids = mrna_ids + lncrna_ids
    n_genes_total = len(all_gene_ids)

    # 确保不超过配置的总基因数
    if n_genes_total > config.N_GENES_TOTAL:
        all_gene_ids = all_gene_ids[:config.N_GENES_TOTAL]
        n_genes_total = config.N_GENES_TOTAL

    # 生成样本ID
    tumor_samples = [f'TCGA-LIHC-{i:04d}-01A' for i in range(config.N_SAMPLES_TUMOR)]
    normal_samples = [f'TCGA-LIHC-{i:04d}-11A' for i in range(config.N_SAMPLES_NORMAL)]

    all_samples = tumor_samples + normal_samples

    # 生成表达矩阵
    print(f"\n生成表达矩阵:")
    print(f"  - 基因数: {n_genes_total:,}")
    print(f"  - 肿瘤样本: {len(tumor_samples)}")
    print(f"  - 正常样本: {len(normal_samples)}")

    expression_matrix = pd.DataFrame(index=all_gene_ids, columns=all_samples)

    # 为每个基因生成表达值
    for gene_id in all_gene_ids:
        # 判断是否是差异表达基因
        is_differential = gene_id in lncrna_ids and \
                        np.random.random() < (config.N_DIFFERENTIAL / len(lncrna_ids))

        if is_differential:
            # 差异表达: 肿瘤和正常组均值不同
            direction = np.random.choice([1, -1])

            # 肿瘤组
            tumor_expr = np.random.normal(
                loc=config.TUMOR_MEAN + direction * 2,  # 2倍差异
                scale=config.TUMOR_STD,
                size=len(tumor_samples)
            )

            # 正常组
            normal_expr = np.random.normal(
                loc=config.NORMAL_MEAN,
                scale=config.NORMAL_STD,
                size=len(normal_samples)
            )

            expr_values = np.concatenate([tumor_expr, normal_expr])

        else:
            # 非差异表达: 两组相同分布
            expr_values = np.random.normal(
                loc=config.NORMAL_MEAN,
                scale=config.NORMAL_STD,
                size=len(all_samples)
            )

        # 确保表达值非负
        expr_values = np.maximum(expr_values, 0)

        expression_matrix.loc[gene_id] = expr_values

    # 生成样本注释表
    pheno_data = pd.DataFrame({
        'sample_id': all_samples,
        'sample_type': ['Tumor'] * len(tumor_samples) + ['Normal'] * len(normal_samples)
    })

    # 保存数据
    expr_file = os.path.join(config.DATA_DIR, 'synthetic_expression_matrix.csv')
    pheno_file = os.path.join(config.DATA_DIR, 'synthetic_phenotype.csv')

    os.makedirs(config.DATA_DIR, exist_ok=True)

    expression_matrix.to_csv(expr_file)
    pheno_data.to_csv(pheno_file, index=False)

    print(f"\n✓ 模拟数据生成完成:")
    print(f"  - 表达矩阵: {expr_file}")
    print(f"  - 样本注释: {pheno_file}")

    # 统计信息
    print(f"\n数据统计:")
    print(f"  - 总基因数: {n_genes_total:,}")
    print(f"  - lncRNA数: {len(lncrna_ids):,}")
    print(f"  - mRNA数: {len([g for g in all_gene_ids if g in mrna_ids]):,}")
    print(f"  - 差异表达基因(模拟): 约{config.N_DIFFERENTIAL}")

    return expr_file, pheno_file


def main():
    """主执行函数"""

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='lncRNA生物标志物发现流程 - TCGA-LIHC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用模拟数据运行完整流程
  python main_pipeline.py --data_type synthetic

  # 使用真实TCGA数据
  python main_pipeline.py --data_type real \\
      --expr data/expression.csv \\
      --phenotype data/phenotype.csv

  # 指定分类器类型
  python main_pipeline.py --data_type synthetic --classifier rf

  # 跳过某些步骤
  python main_pipeline.py --data_type synthetic --skip-visualization
        """
    )

    parser.add_argument(
        '--data_type',
        type=str,
        choices=['synthetic', 'real'],
        default='synthetic',
        help='数据类型: synthetic(模拟) 或 real(真实TCGA)'
    )

    parser.add_argument(
        '--expr',
        type=str,
        help='表达矩阵文件路径'
    )

    parser.add_argument(
        '--phenotype',
        type=str,
        help='样本注释文件路径'
    )

    parser.add_argument(
        '--classifier',
        type=str,
        choices=['svm', 'rf'],
        default=config.CLASSIFIER,
        help=f'分类器类型 (默认: {config.CLASSIFIER})'
    )

    parser.add_argument(
        '--n_biomarkers',
        type=int,
        default=config.N_SELECTED_FEATURES,
        help=f'选择lncRNA标志物数量 (默认: {config.N_SELECTED_FEATURES})'
    )

    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='跳过可视化步骤'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='打印详细信息'
    )

    args = parser.parse_args()

    # 打印开始信息
    print("\n" + "=" * 70)
    print(" "*15 + "lncRNA生物标志物发现流程")
    print("=" * 70)
    print(f"\n项目: {config.EXPERIMENT_METADATA['project']}")
    print(f"目标: {config.EXPERIMENT_METADATA['target_cancer']}")
    print(f"标志物类型: {config.EXPERIMENT_METADATA['biomarker_type']}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # 步骤0: 准备数据
    # ========================================================================

    if args.data_type == 'synthetic':
        # 生成模拟数据
        expr_file, pheno_file = generate_synthetic_tcga_data()
    else:
        # 使用真实数据
        if not args.expr or not os.path.exists(args.expr):
            print(f"\n错误: 表达矩阵文件不存在: {args.expr}")
            sys.exit(1)

        expr_file = args.expr
        pheno_file = args.phenotype if args.phenotype and os.path.exists(args.phenotype) else None

        print(f"\n使用真实数据:")
        print(f"  - 表达矩阵: {expr_file}")
        if pheno_file:
            print(f"  - 样本注释: {pheno_file}")
        else:
            print(f"  - 样本注释: 将从TCGA barcode自动推断")

    # ========================================================================
    # 步骤1-2: 数据预处理和lncRNA过滤
    # ========================================================================

    print("\n" + "=" * 70)
    print("阶段1: 数据预处理和lncRNA注释")
    print("=" * 70)

    # 步骤1: 预处理
    preprocessor = TCGAPreprocessor(verbose=args.verbose)
    tumor_data, normal_data, combined_data = preprocessor.run_preprocessing_pipeline(
        expr_file, pheno_file
    )

    # 步骤2: lncRNA过滤
    annotator = LncRNAAnnotator(verbose=args.verbose)
    lncrna_matrix = annotator.run_annotation_pipeline(combined_data, save_results=True)

    # 分离lncRNA的肿瘤和正常数据
    lncrna_tumor = lncrna_matrix[tumor_data.columns]
    lncrna_normal = lncrna_matrix[normal_data.columns]

    # ========================================================================
    # 步骤3: 差异表达分析
    # ========================================================================

    print("\n" + "=" * 70)
    print("阶段2: 差异表达分析")
    print("=" * 70)

    de_analyzer = DifferentialExpressionAnalyzer(verbose=args.verbose)
    de_results = de_analyzer.identify_differential_expression(
        lncrna_tumor, lncrna_normal
    )

    # 保存DE结果
    de_analyzer.save_results()

    # 提取显著差异lncRNA
    significant_lncrnas = de_results[de_results['significant']]['gene_id'].tolist()

    print(f"\n✓ 发现{len(significant_lncrnas)}个显著差异表达lncRNA")

    # ========================================================================
    # 步骤4: 特征选择
    # ========================================================================

    print("\n" + "=" * 70)
    print("阶段3: 特征选择 - 筛选lncRNA生物标志物")
    print("=" * 70)

    feature_selector = FeatureSelector(verbose=args.verbose)
    selected_biomarkers, biomarker_matrix = feature_selector.run_feature_selection_pipeline(
        lncrna_tumor, lncrna_normal,
        de_genes=significant_lncrnas,
        n_features=args.n_biomarkers,
        method=config.FEATURE_SELECTION_METHOD
    )

    print(f"\n✓ 选择{len(selected_biomarkers)}个lncRNA生物标志物:")
    for i, biomarker in enumerate(selected_biomarkers, 1):
        print(f"  {i}. {biomarker}")

    # ========================================================================
    # 步骤5: 机器学习分类
    # ========================================================================

    print("\n" + "=" * 70)
    print(f"阶段4: 机器学习分类 - {args.classifier.upper()}模型")
    print("=" * 70)

    # 使用选择的生物标志物
    biomarker_tumor = biomarker_matrix[tumor_data.columns.intersection(biomarker_matrix.columns)]
    biomarker_normal = biomarker_matrix[normal_data.columns.intersection(biomarker_matrix.columns)]

    # 训练分类器
    classifier = BiomarkerClassifier(classifier_type=args.classifier, verbose=args.verbose)
    eval_results = classifier.run_classification_pipeline(biomarker_tumor, biomarker_normal)

    # ========================================================================
    # 步骤6: 可视化
    # ========================================================================

    if not args.skip_visualization:
        print("\n" + "=" * 70)
        print("阶段5: 结果可视化")
        print("=" * 70)

        visualizer = BiomarkerVisualizer(verbose=args.verbose)

        # 火山图
        visualizer.plot_volcano(de_results)

        # 热图
        visualizer.plot_heatmap(
            biomarker_matrix,
            tumor_data.columns.tolist(),
            normal_data.columns.tolist()
        )

        # ROC曲线
        visualizer.plot_roc_curve(eval_results)

        # 综合总结图
        visualizer.create_summary_figure(
            de_results,
            biomarker_matrix,
            tumor_data.columns.tolist(),
            normal_data.columns.tolist(),
            eval_results
        )

    # ========================================================================
    # 流程完成 - 总结报告
    # ========================================================================

    print("\n" + "=" * 70)
    print("流程完成! 结果总结")
    print("=" * 70)

    print(f"\n📊 数据统计:")
    print(f"  - 总lncRNA数: {lncrna_matrix.shape[0]:,}")
    print(f"  - 差异表达lncRNA: {len(significant_lncrnas)}")
    print(f"  - 选择标志物: {len(selected_biomarkers)}")

    print(f"\n🎯 分类器性能 ({args.classifier.upper()}):")
    print(f"  - 准确率: {eval_results['accuracy']:.3f}")
    print(f"  - 灵敏度: {eval_results['sensitivity']:.3f}")
    print(f"  - 特异度: {eval_results['specificity']:.3f}")
    if eval_results.get('auc'):
        print(f"  - AUC: {eval_results['auc']:.3f}")

    print(f"\n📁 输出文件:")
    print(f"  - 结果目录: {config.RESULTS_DIR}/")
    print(f"    * 差异表达: {config.OUTPUT_FILES['differential_expression']}")
    print(f"    * 生物标志物: {config.OUTPUT_FILES['biomarkers']}")
    print(f"    * 分类报告: {config.OUTPUT_FILES['classification_report']}")

    if not args.skip_visualization:
        print(f"  - 图表目录: {config.FIGURES_DIR}/")
        print(f"    * 火山图: volcano_plot.png")
        print(f"    * 热图: heatmap_top20.png")
        print(f"    * ROC曲线: roc_curve.png")
        print(f"    * 综合图: summary_figure.png")

    print(f"\n✅ 成功完成lncRNA生物标志物发现流程!")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
