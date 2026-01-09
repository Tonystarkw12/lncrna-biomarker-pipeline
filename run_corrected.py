#!/usr/bin/env python3
"""
修复版: 避免数据泄露的正确流程

关键改进:
1. 先划分训练集和测试集
2. 仅在训练集上进行特征选择
3. 在独立的测试集上评估模型

这样可以得到真实可靠的性能估计
"""

import os
import sys
import pandas as pd
import numpy as np
import gzip
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

from modules.step3_differential_expression import DifferentialExpressionAnalyzer
from modules.step4_feature_selection import FeatureSelector
from modules.step5_classification import BiomarkerClassifier
from modules.step6_visualization import BiomarkerVisualizer

import config


def parse_gtf_for_lncrnas(gtf_file: str) -> set:
    """解析GTF文件提取lncRNA"""
    print("解析GTF文件...")
    lncrna_genes = set()

    with gzip.open(gtf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue

            attributes = fields[8]
            gene_type_match = re.search(r'gene_type "([^"]+)"', attributes)
            gene_id_match = re.search(r'gene_id "ENSG\d+\.?\d*"', attributes)

            if gene_type_match and gene_id_match and gene_type_match.group(1) == 'lncRNA':
                gene_id = re.search(r'ENSG\d+', gene_id_match.group(0)).group(0)
                lncrna_genes.add(gene_id)

    print(f"✓ 提取 {len(lncrna_genes):,} 个lncRNA")
    return lncrna_genes


def load_and_prepare_data(expr_file, clinical_file, lncrna_genes):
    """加载并准备数据"""
    print("\n加载和准备数据...")

    # 加载表达矩阵
    print("  - 加载表达矩阵...")
    expr_df = pd.read_csv(expr_file, sep='\t', compression='gzip', index_col=0)
    expr_df.index = expr_df.index.str.split('.').str[0]

    # 过滤lncRNA
    lncrna_present = [gid for gid in lncrna_genes if gid in expr_df.index]
    expr_lncrna = expr_df.loc[lncrna_present]
    print(f"  ✓ 表达矩阵: {expr_lncrna.shape[0]:,} lncRNA × {expr_lncrna.shape[1]:,} 样本")

    # 加载临床数据
    print("  - 解析样本类型...")
    clinical_df = pd.read_csv(clinical_file, sep='\t', compression='gzip')
    sample_id_col = clinical_df.columns[0]

    tumor_samples = []
    normal_samples = []

    for sid in clinical_df[sample_id_col].values:
        if isinstance(sid, str) and '-' in sid:
            parts = sid.split('-')
            if len(parts) >= 4:
                code = parts[3][:2]
                if code in ['01', '02', '03', '04', '05', '06', '07', '08', '09']:
                    tumor_samples.append(sid)
                elif code in ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19']:
                    normal_samples.append(sid)

    # 匹配样本
    common_tumor = [s for s in tumor_samples if s in expr_lncrna.columns]
    common_normal = [s for s in normal_samples if s in expr_lncrna.columns]

    print(f"  ✓ 匹配样本: {len(common_tumor)} 肿瘤 + {len(common_normal)} 正常")

    # 创建数据框
    tumor_data = expr_lncrna[common_tumor]
    normal_data = expr_lncrna[common_normal]

    # 过滤低表达
    print("  - 过滤低表达lncRNA...")
    mean_expr = pd.concat([tumor_data, normal_data], axis=1).mean(axis=1)
    expr_filtered = pd.concat([tumor_data, normal_data], axis=1).loc[mean_expr >= 1.0]

    tumor_data = tumor_data.loc[expr_filtered.index]
    normal_data = normal_data.loc[expr_filtered.index]

    print(f"  ✓ 保留 {len(tumor_data)} 个表达的lncRNA")

    return tumor_data, normal_data


def main():
    """主流程 - 避免数据泄露"""

    print("\n" + "=" * 70)
    print(" " * 15 + "lncRNA生物标志物发现（修复版）")
    print(" " * 10 + "避免数据泄露的正确流程")
    print("=" * 70)

    # 1. 加载数据
    base_dir = Path(config.PROJECT_DIR)
    expr_file = base_dir / "TCGA-LIHC.star_counts.tsv.gz"
    clinical_file = base_dir / "TCGA-LIHC.clinical.tsv.gz"
    gtf_file = base_dir / "gencode.v36.long_noncoding_RNAs.gtf.gz"

    lncrna_genes = parse_gtf_for_lncrnas(str(gtf_file))
    tumor_data, normal_data = load_and_prepare_data(str(expr_file), str(clinical_file), lncrna_genes)

    # 2. ⭐ 关键改进：先划分训练集和测试集
    print("\n" + "=" * 70)
    print("关键步骤：先划分训练集和测试集")
    print("=" * 70)

    # 合并数据
    all_data = pd.concat([tumor_data, normal_data], axis=1)  # genes x samples
    all_labels = [1] * tumor_data.shape[1] + [0] * normal_data.shape[1]  # 对应样本的标签

    # 划分数据 (分层采样)
    X_train, X_test, y_train, y_test = train_test_split(
        all_data.T,  # 转置为样本×特征
        all_labels,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=all_labels  # 保持类别比例
    )

    print(f"\n数据划分:")
    print(f"  训练集: {len(y_train)} 样本 (肿瘤: {sum(y_train)}, 正常: {len(y_train)-sum(y_train)})")
    print(f"  测试集: {len(y_test)} 样本 (肿瘤: {sum(y_test)}, 正常: {len(y_test)-sum(y_test)})")

    # 保存样本名
    train_sample_names = X_train.index.tolist()
    test_sample_names = X_test.index.tolist()

    # 转换回基因×样本格式
    train_tumor_samples = [train_sample_names[i] for i in range(len(train_sample_names)) if y_train[i] == 1]
    train_normal_samples = [train_sample_names[i] for i in range(len(train_sample_names)) if y_train[i] == 0]
    test_tumor_samples = [test_sample_names[i] for i in range(len(test_sample_names)) if y_test[i] == 1]
    test_normal_samples = [test_sample_names[i] for i in range(len(test_sample_names)) if y_test[i] == 0]

    train_tumor = X_train.T[train_tumor_samples]
    train_normal = X_train.T[train_normal_samples]
    test_tumor = X_test.T[test_tumor_samples]
    test_normal = X_test.T[test_normal_samples]

    # 3. 差异表达分析（仅在训练集上）
    print("\n" + "=" * 70)
    print("步骤1: 差异表达分析（仅训练集）")
    print("=" * 70)

    de_analyzer = DifferentialExpressionAnalyzer(verbose=True)
    de_results = de_analyzer.identify_differential_expression(
        train_tumor, train_normal,
        log2fc_threshold=1.0,
        fdr_threshold=0.05
    )

    significant_genes = de_results[de_results['significant']]['gene_id'].tolist()
    print(f"\n✓ 训练集中发现 {len(significant_genes)} 个显著差异lncRNA")

    # 4. 特征选择（仅在训练集上）⭐ 关键
    print("\n" + "=" * 70)
    print("步骤2: 特征选择（仅训练集）")
    print("=" * 70)

    feature_selector = FeatureSelector(verbose=True)
    selected_biomarkers, _ = feature_selector.run_feature_selection_pipeline(
        train_tumor, train_normal,
        de_genes=significant_genes,
        n_features=20,
        method='lasso'
    )

    print(f"\n✓ 选择 {len(selected_biomarkers)} 个生物标志物")

    # 5. 训练模型（仅用训练集）
    print("\n" + "=" * 70)
    print("步骤3: 训练SVM模型（仅训练集）")
    print("=" * 70)

    # 提取选择的特征
    train_tumor_selected = train_tumor.loc[selected_biomarkers]
    train_normal_selected = train_normal.loc[selected_biomarkers]

    classifier = BiomarkerClassifier(classifier_type='svm', verbose=True)
    classifier.prepare_training_data(train_tumor_selected, train_normal_selected)

    # 训练SVM
    classifier.train_svm()

    # 6. 在测试集上评估（关键步骤）
    print("\n" + "=" * 70)
    print("步骤4: 在独立的测试集上评估")
    print("=" * 70)

    # 准备测试数据
    test_tumor_selected = test_tumor.loc[selected_biomarkers]
    test_normal_selected = test_normal.loc[selected_biomarkers]

    # 合并测试数据
    test_data = pd.concat([test_tumor_selected, test_normal_selected], axis=1)
    test_labels = np.array([1] * len(test_tumor_selected.columns) + [0] * len(test_normal_selected.columns))

    # 标准化（使用训练集的统计量）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # 在训练集上fit
    train_combined = pd.concat([train_tumor_selected, train_normal_selected], axis=1)
    scaler.fit(train_combined.T.values)

    # 转换测试数据
    test_scaled = scaler.transform(test_data.T.values)

    # 预测
    y_pred = classifier.model.predict(test_scaled)
    y_pred_proba = classifier.model.predict_proba(test_scaled)[:, 1]

    # 计算性能指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, zero_division=0)
    recall = recall_score(test_labels, y_pred, zero_division=0)
    f1 = f1_score(test_labels, y_pred, zero_division=0)

    # ROC和AUC
    fpr, tpr, _ = roc_curve(test_labels, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # 混淆矩阵
    cm = confusion_matrix(test_labels, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n📊 测试集性能 (真实性能):")
    print(f"  - 准确率: {accuracy:.3f}")
    print(f"  - 精确率: {precision:.3f}")
    print(f"  - 灵敏度: {recall:.3f}")
    print(f"  - 特异度: {tn/(tn+fp):.3f}")
    print(f"  - F1分数: {f1:.3f}")
    print(f"  - AUC-ROC: {roc_auc:.3f}")

    print(f"\n混淆矩阵:")
    print(f"                预测正常    预测肿瘤")
    print(f"  实际正常       {tn:3d}        {fp:3d}")
    print(f"  实际肿瘤       {fn:3d}        {tp:3d}")

    # 7. 交叉验证（仅在训练集上）
    print("\n" + "=" * 70)
    print("步骤5: 交叉验证（仅在训练集上）")
    print("=" * 70)

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    # 准备训练数据
    X_train_full = train_combined.T.values
    y_train_full = np.array([1] * len(train_tumor_selected.columns) + [0] * len(train_normal_selected.columns))

    # 标准化
    scaler_cv = StandardScaler()
    X_train_scaled = scaler_cv.fit_transform(X_train_full)

    # 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    svm_cv = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', random_state=config.RANDOM_STATE)

    cv_scores = cross_val_score(svm_cv, X_train_scaled, y_train_full, cv=cv, scoring='accuracy')

    print(f"\n✓ 5折交叉验证结果:")
    print(f"  - 各折准确率: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  - 平均准确率: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 8. 总结
    print("\n" + "=" * 70)
    print("总结: 真实性能 vs 过拟合性能")
    print("=" * 70)

    print(f"\n❌ 之前的错误流程（数据泄露）:")
    print(f"   - 测试集准确率: 100% (虚假的完美)")
    print(f"   - AUC: 1.000 (不现实)")
    print(f"   - 原因: 特征选择时看到了测试集")

    print(f"\n✅ 修复后的正确流程:")
    print(f"   - 测试集准确率: {accuracy:.1%}")
    print(f"   - AUC: {roc_auc:.3f}")
    print(f"   - 交叉验证: {cv_scores.mean():.1%} ± {cv_scores.std():.3f}")
    print(f"   - 原因: 测试集完全独立")

    print(f"\n💡 性能解读:")
    if accuracy > 0.95:
        print(f"   - 性能优秀，但需要验证")
        print(f"   - 建议: 在独立数据集上验证")
    elif accuracy > 0.85:
        print(f"   - 性能良好，结果可信")
    else:
        print(f"   - 性能一般，可能需要优化特征或模型")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
