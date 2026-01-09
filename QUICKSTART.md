# 快速开始指南 - lncRNA生物标志物发现流程

## 安装依赖

```bash
# 创建conda环境
conda create -n lncrna_ml python=3.8
conda activate lncrna_ml

# 安装Python包
pip install -r requirements.txt
```

## 运行完整流程

### 使用真实TCGA-LIHC数据 (推荐)

**⭐ 重要**: 请使用 `run_corrected.py` 脚本，该版本修复了数据泄露问题，性能评估更可靠。

```bash
# 确保数据文件在data/目录:
# - TCGA-LIHC.star_counts.tsv.gz
# - TCGA-LIHC.clinical.tsv.gz
# - gencode.v36.long_noncoding_RNAs.gtf.gz

# 运行修复版流程 (避免数据泄露)
python run_corrected.py
```

### 脚本说明

- **`run_corrected.py`** ⭐ (推荐)
  - 修复版流程，避免数据泄露
  - 先划分训练/测试集，再进行特征选择
  - 性能评估更真实可靠
  - 交叉验证准确率: 98.5% ± 0.9%

- **`run_real_data.py`** (旧版，仅供参考)
  - 存在数据泄露问题
  - 特征选择使用了全部数据
  - 导致虚高的性能指标

## 查看结果

### 结果文件

所有结果保存在 `results/` 目录:

- `differential_expression.csv`: 差异表达分析结果
  - 1,748个显著差异lncRNA (训练集)
  - 包含: gene_id, log2FC, pvalue, qvalue, regulation

- `selected_biomarkers.csv`: 选择的lncRNA标志物
  - 12个关键lncRNA (Lasso选择)
  - 包含: Lasso系数, RF重要性排名

- `classification_report.txt`: 模型性能报告
  - 测试集性能
  - 交叉验证结果 (5折)
  - 混淆矩阵

- `sample_predictions.csv`: 所有样本的预测结果
- `trained_model.pkl`: 训练好的SVM模型

### 可视化图表

所有图表保存在 `figures/` 目录:

- `volcano_plot.png`: 火山图
  - 展示所有lncRNA的差异表达格局
  - 1,748个显著差异lncRNA (红色上调，蓝色下调)

- `heatmap_top20.png`: 热图
  - Top 20标志物的表达模式
  - 424个样本的层次聚类

- `roc_curve.png`: ROC曲线
  - AUC = 1.000
  - 展示分类器的诊断能力

- `summary_figure.png`: 综合总结图
  - 火山图 + 热图 + ROC曲线
  - 适合用于报告和展示

## 调整参数

编辑 `config.py` 文件可调整所有参数:

```python
# 差异表达阈值
LOG2FC_THRESHOLD = 1.0  # Log2折叠变化阈值 (|>1| 表示2倍差异)
FDR_THRESHOLD = 0.05    # FDR显著性阈值

# 特征选择
N_SELECTED_FEATURES = 20  # 选择标志物数量
FEATURE_SELECTION_METHOD = 'lasso'  # 特征选择方法

# 机器学习
CLASSIFIER = 'svm'  # 分类器类型 ('svm' 或 'rf')
TEST_SIZE = 0.2     # 测试集比例 (0.2 = 20%测试集)
SVM_C = 1.0         # SVM正则化参数
```

## 性能指标解读

### 主要指标

- **交叉验证准确率** (最可靠): 98.5% ± 0.9%
  - 基于训练集的5折交叉验证
  - 反映模型的真实泛化能力
  - **这是最重要的性能指标**

- **测试集准确率**: 100%
  - 基于独立测试集 (85个样本)
  - ⚠️ 测试集正常样本仅10个，规模较小
  - 仅供参考

- **AUC-ROC**: 1.000
  - 完美的分类性能
  - ⚠️ 受测试集规模限制，需谨慎解读

### 为什么AUC=1.0?

这是**正常的**，因为:
1. 测试集规模较小 (仅10个正常样本)
2. lncRNA差异表达显著 (最大Log2FC > 4)
3. 更可靠的指标是交叉验证 (98.5% ± 0.9%)
4. 模型在训练集和测试集上表现一致，表明没有过拟合

## 常见问题

**Q: 如何增加测试集大小?**
```python
# 在config.py中修改:
TEST_SIZE = 0.4  # 使用40%的数据作为测试集
```

**Q: 如何选择更多生物标志物?**
```python
# 在config.py中修改:
N_SELECTED_FEATURES = 30  # 选择30个标志物
```

**Q: 使用Random Forest代替SVM?**
```python
# 在run_corrected.py中修改:
classifier = BiomarkerClassifier(classifier_type='rf', verbose=True)
```

**Q: 为什么交叉验证准确率是98.5%而不是100%?**
这是**真实的性能估计**。测试集100%是因为测试集样本较少，而交叉验证在更大的数据集上评估，更能反映真实泛化能力。

## 数据要求

如需使用自己的TCGA数据，确保:

1. **表达矩阵** (如 `TCGA-LIHC.star_counts.tsv.gz`):
   - 格式: 基因 × 样本 (TSV, gzip压缩)
   - 第一列: Ensembl Gene ID (如 ENSG00000225383.5)
   - 其余列: 样本表达值 (log2 counts)
   - 表头: TCGA barcode (如 TCGA-DD-A73C-01A)

2. **临床数据** (如 `TCGA-LIHC.clinical.tsv.gz`):
   - 格式: 样本 × 特征 (TSV, gzip压缩)
   - 第一列: TCGA barcode
   - 必须包含样本类型信息 (在barcode中)

3. **lncRNA注释** (如 `gencode.v36.long_noncoding_RNAs.gtf.gz`):
   - GENCODE GTF格式 (gzip压缩)
   - 包含基因类型注释 (gene_type "lncRNA")

## 模块化使用

可以导入单个模块在Jupyter/脚本中使用:

```python
import sys
sys.path.append('.')

from modules.step3_differential_expression import DifferentialExpressionAnalyzer
from modules.step4_feature_selection import FeatureSelector
from modules.step5_classification import BiomarkerClassifier

# 仅运行差异表达分析
de_analyzer = DifferentialExpressionAnalyzer(verbose=True)
de_results = de_analyzer.identify_differential_expression(
    tumor_data, normal_data,
    log2fc_threshold=1.0,
    fdr_threshold=0.05
)

# 仅运行特征选择
feature_selector = FeatureSelector(verbose=True)
biomarkers, _ = feature_selector.run_feature_selection_pipeline(
    tumor_data, normal_data,
    de_genes=de_results[de_results['significant']]['gene_id'].tolist(),
    n_features=20,
    method='lasso'
)
```

## 下一步分析

### 实验验证
1. **qRT-PCR**: 验证Top 5-10 lncRNA在独立样本中的表达
2. **独立队列**: 在其他TCGA队列 (如TCGA-CHOL) 中验证
3. **功能研究**: 敲低/过表达关键lncRNA研究生物学功能

### 深入分析
1. **生存分析**: 评估lncRNA的预后价值 (KM曲线, Cox回归)
2. **通路富集**: 了解相关信号通路 (GO, KEGG)
3. **网络分析**: 构建lncRNA-mRNA共表达网络
4. **上游调控**: 预测转录因子 (motif分析)

### 临床转化
1. **液体活检**: 检测血清/血浆中lncRNA
2. **早期诊断**: 识别早期(I期)肝癌标志物
3. **治疗靶点**: 探索治疗潜力

## 引用

如果使用本流程，请引用:

```
TCGA Research Network. Comprehensive and Integrative Genomic
Characterization of Hepatocellular Carcinoma. Cell. 2017.

GENCODE Consortium. The GENCODE Encyclopedia of human
genomic features. Nature 2019.
```

---

**祝研究顺利! 🎉**

**最后更新**: 2025-01-09
**版本**: v2.0 (修复数据泄露版本)
