"""
配置文件 - lncRNA Biomarker Discovery Pipeline

包含所有可调参数，方便实验复现和参数优化
"""

import os

# ============================================================================
# 路径配置
# ============================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
MODULES_DIR = os.path.join(PROJECT_DIR, 'modules')

# 创建必要的目录
for dir_path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# 数据配置
# ============================================================================
# TCGA样本类型代码 (从barcode解析)
# https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables
SAMPLE_TYPE_CODES = {
    '01': 'Primary Tumor',
    '11': 'Solid Tissue Normal',
    '10': 'Blood Derived Normal',
    '06': 'Metastatic'
}

# 保留的样本类型 (用于分类)
TUMOR_CODES = ['01', '06']  # 原发肿瘤 + 转移
NORMAL_CODES = ['11']       # 实体正常组织

# ============================================================================
# lncRNA识别配置
# ============================================================================
# Ensembl基因ID规则: lncRNA通常ID编号 > 60000
# 基于GENCODE v41: 蛋白编码基因约20000个，lncRNA约50000个
ENSEMBL_LNCRNA_ID_START = 60000
ENSEMBL_LNCRNA_ID_END = 100000

# 模拟已知lncRNA列表的参数
SYNTHETIC_LNCRNA_FRACTION = 0.3  # 假设30%的基因是lncRNA
SYNTHETIC_LNCRNA_N = 5000        # 合成数据中lncRNA的数量

# ============================================================================
# 数据预处理配置
# ============================================================================
# 表达值转换
LOG_TRANSFORM = True        # 是否进行log2转换
LOG_PSEUDOCOUNT = 1         # log(x + pseudocount)
NORMALIZE = 'zscore'        # 标准化方法: 'zscore', 'minmax', 或 None

# 低表达过滤
MIN_CPM = 1                 # 最小CPM值 (Counts Per Million)
MIN_SAMPLES_EXPRESSED = 0.2  # 至少在20%样本中表达

# ============================================================================
# 差异表达分析配置
# ============================================================================
# 统计检验方法: 'wilcoxon' 或 'ttest'
DE_METHOD = 'wilcoxon'

# 显著性阈值
LOG2FC_THRESHOLD = 1.0      # |log2FC| > 1 (至少2倍差异)
FDR_THRESHOLD = 0.05        # FDR < 0.05
PVALUE_THRESHOLD = 0.01     # 原始p值阈值

# FDR校正方法
FDR_METHOD = 'fdr_bh'       # Benjamini-Hochberg

# ============================================================================
# 特征选择配置
# ============================================================================
# 特征选择方法: 'lasso', 'rf', 'variance_threshold', 或 'univariate'
FEATURE_SELECTION_METHOD = 'lasso'

# Lasso回归参数
LASSO_ALPHA_RANGE = [0.001, 0.01, 0.1, 1.0, 10.0]  # 正则化强度搜索范围
LASSO_CV_FOLDS = 5          # 交叉验证折数
LASSO_MAX_ITER = 10000      # 最大迭代次数

# Random Forest参数
RF_N_ESTIMATORS = 100       # 树的数量
RF_MAX_FEATURES = 'sqrt'    # 每棵树考虑的最大特征数
RF_RANDOM_STATE = 42

# 选择特征数量
N_SELECTED_FEATURES = 20    # 最终选择的标志物数量

# ============================================================================
# 机器学习分类配置
# ============================================================================
# 分类器: 'svm', 'rf', 或 'ensemble'
CLASSIFIER = 'svm'

# 数据划分
TEST_SIZE = 0.2             # 测试集比例
VALIDATION_SIZE = 0.2       # 验证集比例(用于early stopping)
STRATIFY = True             # 是否分层采样(保持类别平衡)

# 随机种子
RANDOM_STATE = 42

# SVM参数
SVM_KERNEL = 'rbf'          # 'linear', 'poly', 'rbf', 'sigmoid'
SVM_C = 1.0                 # 正则化参数(越大越不正则化)
SVM_GAMMA = 'scale'         # 'scale', 'auto', 或float
SVM_CLASS_WEIGHT = 'balanced'  # 处理类别不平衡
SVM_PROBABILITY = True      # 是否输出概率(用于ROC)

# Random Forest分类器参数
RF_CLF_N_ESTIMATORS = 100
RF_CLF_MAX_DEPTH = 10
RF_CLF_MIN_SAMPLES_SPLIT = 2
RF_CLF_MIN_SAMPLES_LEAF = 1

# 交叉验证
CV_FOLDS = 5                # 用于模型评估的交叉验证折数

# ============================================================================
# 可视化配置
# ============================================================================
# 图表样式
FIGURE_DPI = 300            # 高分辨率用于发表
FIGURE_FORMAT = 'png'       # 'png', 'pdf', 'svg'
COLOR_SCHEME = 'viridis'    # 'viridis', 'plasma', 'coolwarm'

# 火山图参数
VOLCANO_FDR = 0.05          # 标注显著基因的FDR阈值
VOLCANO_LOG2FC = 1.0        # 标注显著基因的log2FC阈值
VOLCANO_TOP_N = 10          # 标注top N个基因名

# 热图参数
HEATMAP_N_BIOMARKERS = 20   # 显示的标志物数量
HEATMAP_CLUSTER_SAMPLES = True
HEATMAP_CLUSTER_GENES = True
HEATMAP_STANDARD_SCALE = 0  # 0:按行标准化, 1:按列标准化, None:不标准化

# ROC曲线参数
ROC_CV_FOLDS = 5            # 用于绘制CV confidence interval的折数
ROC_SHOW_CI = True          # 是否显示置信区间

# ============================================================================
# 合成数据配置(用于演示)
# ============================================================================
N_SAMPLES_TUMOR = 150       # 肿瘤样本数
N_SAMPLES_NORMAL = 50       # 正常样本数
N_GENES_TOTAL = 15000       # 总基因数
N_DIFFERENTIAL = 500        # 差异表达基因数

# 合成数据表达值分布参数
TUMOR_MEAN = 6.0            # log2表达值均值
TUMOR_STD = 2.0
NORMAL_MEAN = 5.0
NORMAL_STD = 2.0

# mRNA ID范围 (用于生成合成数据)
SYNTHETIC_MRNA_ID_END = ENSEMBL_LNCRNA_ID_START  # 60000

# ============================================================================
# 输出配置
# ============================================================================
VERBOSE = True              # 是否打印详细信息
SAVE_INTERMEDIATE = True    # 是否保存中间结果

# 文件名模板
OUTPUT_FILES = {
    'differential_expression': 'differential_expression.csv',
    'biomarkers': 'selected_biomarkers.csv',
    'classification_report': 'classification_report.txt',
    'model': 'trained_model.pkl',
    'predictions': 'sample_predictions.csv'
}

# ============================================================================
# 实验元数据(用于记录)
# ============================================================================
EXPERIMENT_METADATA = {
    'project': 'TCGA-LIHC lncRNA Biomarker Discovery',
    'data_source': 'UCSC Xena Browser / Synthetic',
    'target_cancer': 'Liver Hepatocellular Carcinoma (LIHC)',
    'biomarker_type': 'Long non-coding RNA (lncRNA)',
    'classification_task': 'Tumor vs Normal',
    'created_by': 'Bioinformatics Ph.D. Student'
}
