# lncRNA Biomarker Discovery Pipeline for TCGA-LIHC

## 项目概述

本项目实现了一个完整的、生产级的机器学习流程，用于识别区分肝细胞癌(LIHC)肿瘤样本与正常样本的**长链非编码RNA (lncRNA)**生物标志物。

### ⚠️ 重要更新

**版本历史:**
- **v2.0 (当前)**: 修复数据泄露问题，正确的训练/测试划分
  - 先划分数据集，再进行特征选择
  - 交叉验证准确率: **98.5% ± 0.9%**
  - AUC-ROC: 1.000 (测试集规模较小，仅供参考)

- **v1.0**: 初始版本（存在数据泄露）
  - 特征选择使用了全部数据
  - 导致虚假的完美性能 (AUC=1.0)

### 生物学背景

#### 为什么选择lncRNA？
- **调控功能**: lncRNA在表观遗传调控、转录及转录后调控中发挥关键作用
- **疾病关联**: 越来越多的证据表明lncRNA在癌症发生、发展中起重要作用
- **生物标志物潜力**: lncRNA在组织和体液中稳定表达，是理想的诊断/预后标志物
- **液体活检应用**: lncRNA可作为血液cfRNA检测的潜在靶点

#### 为什么选择TCGA-LIHC？
- **大规模**: TCGA提供超过300个LIHC样本的多组学数据
- **标准化**: 统一的处理流程和质量控制
- **临床注释**: 包含完整的生存、分期等临床信息
- **研究价值**: 肝癌是高发癌症，需要新型诊断标志物

## 项目结构

```
lncrna_biomarker_pipeline/
├── config.py                        # 全局配置参数
├── run_corrected.py                # ⭐ 主执行脚本 (修复版，避免数据泄露)
├── run_real_data.py                # 旧版脚本 (仅作参考)
├── main_pipeline.py                # 模拟数据脚本
├── requirements.txt                 # Python依赖包
├── USAGE_SUMMARY.md                 # 项目总结和结果
├── QUICKSTART.md                    # 快速开始指南
├── README.md                        # 本文档
│
├── modules/                         # 功能模块
│   ├── __init__.py
│   ├── step1_preprocessing.py      # 数据预处理
│   ├── step2_annotation_filter.py  # lncRNA注释过滤
│   ├── step3_differential_expression.py  # 差异表达分析
│   ├── step4_feature_selection.py  # 特征选择 (Lasso/RF)
│   ├── step5_classification.py     # 机器学习分类 (SVM/RF)
│   └── step6_visualization.py      # 可视化
│
├── data/                            # 数据目录
│   ├── TCGA-LIHC.star_counts.tsv.gz      # 基因表达 (log2 counts)
│   ├── TCGA-LIHC.clinical.tsv.gz         # 临床数据
│   └── gencode.v36.long_noncoding_RNAs.gtf.gz  # lncRNA注释
│
├── results/                         # 分析结果 ✅
│   ├── differential_expression.csv         # 差异表达结果
│   ├── selected_biomarkers.csv             # 选择的生物标志物
│   ├── classification_report.txt           # 模型性能报告
│   ├── sample_predictions.csv              # 样本预测
│   └── trained_model.pkl                   # 训练好的SVM模型
│
└── figures/                         # 可视化图表 ✅
    ├── volcano_plot.png                      # 火山图
    ├── heatmap_top20.png                     # 热图
    ├── roc_curve.png                         # ROC曲线
    └── summary_figure.png                    # 综合总结图
```

## 方法学原理

### 1. 数据预处理
- **TCGA条形码解析**: 利用样本ID中第14-15位识别组织类型
  - `01A-09B`: 原发肿瘤
  - `11A-11B`: 实体正常组织
- **对数转换**: 数据已是log2(counts+1)转换
- **标准化**: Z-score标准化
- **低表达过滤**: 保留平均表达≥1的lncRNA

### 2. lncRNA识别
- 从GENCODE v36 GTF文件提取lncRNA注释
- 去除Ensembl ID版本号 (如ENSG00000225383.5 → ENSG00000225383)
- 过滤表达矩阵仅保留lncRNA

### 3. 差异表达分析
- **统计检验**: Wilcoxon秩和检验 (非参数，适合非正态分布)
- **多重假设检验校正**: Benjamini-Hochberg FDR校正
- **阈值标准**:
  - |Log2FC| > 1 (至少2倍差异)
  - FDR < 0.05

### 4. 特征选择 (关键改进)
- **避免数据泄露**: 仅在训练集上进行特征选择
- **为什么使用Lasso?**
  - 稀疏性: L1正则化将不重要特征的系数压缩为0
  - 可解释性: 自动选择最重要的特征
  - 共线性处理: 有效处理相关特征
  - 高维数据: 适合特征数>>样本数的场景

### 5. 分类模型
- **算法**: 支持向量机 (SVM) with RBF核
- **为什么选择SVM?**
  - 高维数据: 基因表达数据特征数>>样本数
  - 小样本: 即使样本量较小也能获得良好泛化
  - 核技巧: RBF核可以捕获复杂的非线性关系

### 6. 模型评估
- **交叉验证**: 5折分层交叉验证
- **测试集**: 20%独立测试集 (仅用于最终评估)
- **评估指标**:
  - 准确率 (Accuracy)
  - 精确率 (Precision)
  - 召回率/灵敏度 (Recall/Sensitivity)
  - 特异度 (Specificity)
  - F1分数 (F1-score)
  - AUC-ROC

## 主要结果

### 数据概况
```
数据集: TCGA-LIHC
- 总样本: 424 (374 肿瘤 + 50 正常)
- 训练集: 339 (299 肿瘤 + 40 正常)
- 测试集: 85 (75 肿瘤 + 10 正常)
- lncRNA: 5,893个 (表达基因)
- 差异表达lncRNA: 1,748个 (FDR<0.05, |Log2FC|>1)
```

### 生物标志物
通过Lasso回归选择了**12个关键lncRNA生物标志物**:

| 排名 | lncRNA ID | Log2FC | 调控方向 | Lasso系数 |
|------|-----------|---------|---------|-----------|
| 1 | ENSG00000228842 | -3.56 | 下调 | 0.0389 |
| 2 | ENSG00000225383 | +3.18 | 上调 | 0.0366 |
| 3 | ENSG00000233542 | +2.82 | 上调 | 0.0261 |
| 4 | ENSG00000263412 | +2.76 | 上调 | 0.0180 |
| 5 | ENSG00000260942 | +2.72 | 上调 | 0.0130 |

### 模型性能

| 指标 | 训练集 | 测试集 | 5折交叉验证 |
|------|--------|--------|-------------|
| 准确率 | 98.9% | 100% | **98.5% ± 0.9%** |
| AUC-ROC | 1.000 | 1.000 | N/A |
| 灵敏度 | - | 100% | N/A |
| 特异度 | - | 100% | N/A |

**性能解读:**
- ✅ **交叉验证98.5%**: 这是**最可靠的性能估计**
- ⚠️ 测试集100%: 由于测试集正常样本仅10个，仅供参考
- ✅ 模型稳定性好 (标准差仅0.9%)

## 安装

```bash
# 创建conda环境
conda create -n lncrna_ml python=3.8
conda activate lncrna_ml

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 运行完整流程 (修复版)

```bash
# 激活环境
conda activate lncrna_ml

# 运行修复版流程 (避免数据泄露)
python run_corrected.py
```

### 查看结果

**结果文件** (results/):
- `differential_expression.csv` - 1,748个差异表达lncRNA
- `selected_biomarkers.csv` - 12个生物标志物详情
- `classification_report.txt` - 模型性能报告
- `sample_predictions.csv` - 所有样本的预测结果

**可视化图表** (figures/):
- `volcano_plot.png` - 差异表达火山图
- `heatmap_top20.png` - Top 20生物标志物热图
- `roc_curve.png` - ROC曲线
- `summary_figure.png` - 综合总结图

## 参数调整

所有参数在 `config.py` 中可调整:

```python
# 差异表达阈值
LOG2FC_THRESHOLD = 1.0      # Log2FC阈值
FDR_THRESHOLD = 0.05        # FDR显著性阈值

# 特征选择
N_SELECTED_FEATURES = 20    # 选择标志物数量
FEATURE_SELECTION_METHOD = 'lasso'

# 机器学习
CLASSIFIER = 'svm'          # 'svm' 或 'rf'
TEST_SIZE = 0.2             # 测试集比例
SVM_C = 1.0                 # SVM正则化参数
```

## 常见问题

**Q: 为什么AUC=1.0?**
A: 测试集正常样本仅10个，规模较小。更可靠的指标是5折交叉验证(98.5%)。

**Q: 如何增加测试集大小?**
A: 在`config.py`中修改`TEST_SIZE = 0.4`，使用40%的数据作为测试集。

**Q: 模型是否过拟合?**
A: 交叉验证结果显示模型稳定（低标准差），且在独立测试集上表现良好。

**Q: 如何选择更多生物标志物?**
A: 修改`N_SELECTED_FEATURES = 30`。

## 技术栈

- **Python 3.8+**
- **数据处理**: pandas, numpy, scipy
- **机器学习**: scikit-learn
- **可视化**: matplotlib, seaborn
- **统计**: statsmodels

## 数据来源

本分析使用以下公开数据:

1. **基因表达**: TCGA-LIHC STAR Counts (UCSC Xena)
2. **临床数据**: TCGA-LIHC Clinical Data (GDC)
3. **lncRNA注释**: GENCODE v36 (EBI)

## 后续工作建议

### 实验验证
1. **qRT-PCR**: 验证Top 5-10 lncRNA的表达差异
2. **独立队列**: 在其他TCGA队列或独立数据集上验证
3. **功能研究**: 敲低/过表达关键lncRNA研究生物学功能

### 深入分析
1. **生存分析**: 评估lncRNA的预后价值
2. **通路富集**: 了解相关信号通路和生物学功能
3. **网络分析**: 构建lncRNA-mRNA共表达网络

### 临床转化
1. **液体活检**: 检测血清/血浆中lncRNA
2. **早期诊断**: 识别早期肝癌标志物
3. **治疗靶点**: 探索治疗潜力

## 学术价值

本项目展示了以下技能:
- ✅ **生物信息学**: TCGA数据处理、差异表达分析
- ✅ **机器学习**: 特征工程、模型训练、性能评估
- ✅ **编程能力**: Python, 模块化设计, 生产级代码
- ✅ **科研思维**: 实验设计、问题诊断、结果解读

**特别强调:**
- 正确处理训练/测试集划分
- 避免数据泄露
- 使用交叉验证获得可靠性能估计

## 引用

如果您在研究中使用了本流程，请引用:

```
TCGA Research Network. Comprehensive and Integrative Genomic
Characterization of Hepatocellular Carcinoma. Cell. 2017.

GENCODE Consortium. The GENCODE Encyclopedia of human
genomic features. Nature 2019.
```

## 作者

Bioinformatics Ph.D. Student
Lab Rotation Application

## 许可证

MIT License

---

**最后更新**: 2025-01-09
**版本**: v2.0 (修复数据泄露版本)
