"""
模块1: 数据预处理

功能:
1. 加载TCGA基因表达数据和样本注释
2. 解析TCGA barcode识别肿瘤/正常样本
3. 数据清洗和质量控制
4. 表达值转换和标准化

作者: Bioinformatics Ph.D. Student
日期: 2025
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径以导入config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TCGAPreprocessor:
    """
    TCGA数据预处理器

    主要功能:
    - 加载表达矩阵和样本信息
    - 解析TCGA barcode样本类型
    - 分离肿瘤和正常样本
    - 表达值转换和标准化
    """

    def __init__(self, verbose=True):
        """
        初始化预处理器

        Parameters:
        -----------
        verbose : bool
            是否打印详细信息
        """
        self.verbose = verbose
        self.expression_data = None
        self.sample_info = None
        self.tumor_samples = []
        self.normal_samples = []

    def load_data(self, expression_file: str, phenotype_file: str = None) -> None:
        """
        加载表达数据和样本信息

        Parameters:
        -----------
        expression_file : str
            表达矩阵文件路径 (genes x samples)
        phenotype_file : str, optional
            样本注释文件路径
        """
        if self.verbose:
            print("=" * 60)
            print("步骤1: 加载TCGA-LIHC数据")
            print("=" * 60)

        # 加载表达矩阵
        try:
            self.expression_data = pd.read_csv(expression_file, index_col=0, sep='\t')
            if self.verbose:
                print(f"\n✓ 成功加载表达矩阵")
                print(f"  - 基因数量: {self.expression_data.shape[0]:,}")
                print(f"  - 样本数量: {self.expression_data.shape[1]:,}")
        except Exception as e:
            raise Exception(f"加载表达数据失败: {str(e)}")

        # 如果有phenotype文件，加载样本信息
        if phenotype_file and os.path.exists(phenotype_file):
            try:
                self.sample_info = pd.read_csv(phenotype_file, sep='\t', index_col=0)
                if self.verbose:
                    print(f"\n✓ 成功加载样本注释")
                    print(f"  - 样本信息: {self.sample_info.shape[0]:,} 个样本")
            except Exception as e:
                if self.verbose:
                    print(f"\n⚠ 警告: 无法加载样本注释 ({str(e)})")
                print("  将从barcode自动推断样本类型")

    def parse_tcga_barcode(self, sample_id: str) -> Dict[str, str]:
        """
        解析TCGA barcode识别样本类型

        TCGA barcode格式: TCGA-XX-XXXX-XXXX-XXXX-XX
        关键位置:
          - 项目代码: 前4个字符 (如TCGA)
          - 样本类型: 第14-15位字符
            * 01-09: 肿瘤样本
            * 10-19: 正常样本
            * 20-29: 对照样本

        参考: https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables

        Parameters:
        -----------
        sample_id : str
            TCGA样本ID

        Returns:
        --------
        dict : 包含解析信息的字典
        """
        # 移除可能的版本号后缀 (如TCGA-XX-XXXX-01A-01R-0001-01)
        base_id = sample_id.split('-')[0:4]
        base_id = '-'.join(base_id)

        if len(base_id) < 4:
            return {'sample_type': 'Unknown', 'category': 'Unknown'}

        # 第14-15位是样本类型代码
        sample_type_code = base_id.split('-')[3][:2]

        # 判断样本类别
        if sample_type_code in config.TUMOR_CODES:
            category = 'Tumor'
        elif sample_type_code in config.NORMAL_CODES:
            category = 'Normal'
        else:
            category = 'Other'

        return {
            'sample_type': config.SAMPLE_TYPE_CODES.get(sample_type_code, 'Unknown'),
            'code': sample_type_code,
            'category': category
        }

    def separate_samples(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        分离肿瘤和正常样本

        基于TCGA barcode或phenotype信息识别样本类型

        Returns:
        --------
        tumor_data : pd.DataFrame
            肿瘤样本表达矩阵
        normal_data : pd.DataFrame
            正常样本表达矩阵
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("步骤2: 识别并分离肿瘤/正常样本")
            print("=" * 60)

        # 获取所有样本ID
        all_samples = self.expression_data.columns.tolist()

        # 如果有phenotype信息，优先使用
        if self.sample_info is not None:
            # 尝试匹配样本
            tumor_cols = []
            normal_cols = []

            for sample in all_samples:
                # 简化样本ID以匹配(可能版本不同)
                sample_short = '-'.join(sample.split('-')[:4]) + '-01'

                if sample_short in self.sample_info.index:
                    sample_type = self.sample_info.loc[sample_short, 'sample_type']

                    # 根据样本类型列判断
                    if isinstance(sample_type, str):
                        if 'Tumor' in sample_type or 'Primary' in sample_type:
                            tumor_cols.append(sample)
                        elif 'Normal' in sample_type:
                            normal_cols.append(sample)

            if tumor_cols and normal_cols:
                self.tumor_samples = tumor_cols
                self.normal_samples = normal_cols
        else:
            # 从barcode解析
            tumor_cols = []
            normal_cols = []

            for sample in all_samples:
                parsed = self.parse_tcga_barcode(sample)

                if parsed['category'] == 'Tumor':
                    tumor_cols.append(sample)
                elif parsed['category'] == 'Normal':
                    normal_cols.append(sample)

            self.tumor_samples = tumor_cols
            self.normal_samples = normal_cols

        # 创建分离的数据框
        tumor_data = self.expression_data[self.tumor_samples]
        normal_data = self.expression_data[self.normal_samples]

        if self.verbose:
            print(f"\n✓ 样本分类完成:")
            print(f"  - 肿瘤样本: {len(self.tumor_samples)}")
            print(f"  - 正常样本: {len(self.normal_samples)}")
            print(f"  - 其他样本: {len(all_samples) - len(self.tumor_samples) - len(self.normal_samples)}")

            # 示例样本ID
            if len(self.tumor_samples) > 0:
                print(f"\n  肿瘤样本示例: {self.tumor_samples[0]}")
                parsed = self.parse_tcga_barcode(self.tumor_samples[0])
                print(f"    -> 类型: {parsed['sample_type']} (代码: {parsed['code']})")

            if len(self.normal_samples) > 0:
                print(f"  正常样本示例: {self.normal_samples[0]}")
                parsed = self.parse_tcga_barcode(self.normal_samples[0])
                print(f"    -> 类型: {parsed['sample_type']} (代码: {parsed['code']})")

        return tumor_data, normal_data

    def filter_low_expression(self, data: pd.DataFrame,
                              min_cpm: float = None,
                              min_samples: float = None) -> pd.DataFrame:
        """
        过滤低表达基因

        保留在足够样本中有表达量的基因

        Parameters:
        -----------
        data : pd.DataFrame
            表达矩阵 (genes x samples)
        min_cpm : float, optional
            最小CPM值
        min_samples : float, optional
            最小样本比例

        Returns:
        --------
        pd.DataFrame : 过滤后的表达矩阵
        """
        if min_cpm is None:
            min_cpm = config.MIN_CPM
        if min_samples is None:
            min_samples = config.MIN_SAMPLES_EXPRESSED

        if self.verbose:
            print(f"\n" + "=" * 60)
            print("步骤3: 过滤低表达基因")
            print("=" * 60)
            print(f"\n过滤标准:")
            print(f"  - 最小表达值: > {min_cpm}")
            print(f"  - 最小样本比例: > {min_samples*100:.1f}%")

        n_genes_before = data.shape[0]

        # 计算每个基因在多少样本中表达
        expressed_samples = (data > min_cpm).sum(axis=1)
        min_required_samples = int(data.shape[1] * min_samples)

        # 保留符合条件的基因
        mask = expressed_samples >= min_required_samples
        filtered_data = data.loc[mask]

        n_genes_after = filtered_data.shape[0]
        n_removed = n_genes_before - n_genes_after

        if self.verbose:
            print(f"\n✓ 过滤完成:")
            print(f"  - 过滤前基因数: {n_genes_before:,}")
            print(f"  - 过滤后基因数: {n_genes_after:,}")
            print(f"  - 移除基因数: {n_removed:,} ({n_removed/n_genes_before*100:.1f}%)")

        return filtered_data

    def transform_expression(self, tumor_data: pd.DataFrame,
                            normal_data: pd.DataFrame,
                            log_transform: bool = None,
                            normalize: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        表达值转换和标准化

        1. Log2转换: log2(x + 1) 以稳定方差
        2. Z-score标准化: 每基因标准化为均值0,标准差1

        Parameters:
        -----------
        tumor_data : pd.DataFrame
            肿瘤样本表达矩阵
        normal_data : pd.DataFrame
            正常样本表达矩阵
        log_transform : bool, optional
            是否进行log转换
        normalize : str, optional
            标准化方法 ('zscore', 'minmax', None)

        Returns:
        --------
        tumor_transformed : pd.DataFrame
            转换后的肿瘤数据
        normal_transformed : pd.DataFrame
            转换后的正常数据
        """
        if log_transform is None:
            log_transform = config.LOG_TRANSFORM
        if normalize is None:
            normalize = config.NORMALIZE

        if self.verbose:
            print(f"\n" + "=" * 60)
            print("步骤4: 表达值转换和标准化")
            print("=" * 60)

        tumor_transformed = tumor_data.copy()
        normal_transformed = normal_data.copy()

        # Log2转换
        if log_transform:
            if self.verbose:
                print(f"\n✓ Log2转换: log2(x + {config.LOG_PSEUDOCOUNT})")

            tumor_transformed = np.log2(tumor_transformed + config.LOG_PSEUDOCOUNT)
            normal_transformed = np.log2(normal_transformed + config.LOG_PSEUDOCOUNT)

        # 标准化
        if normalize == 'zscore':
            if self.verbose:
                print(f"✓ Z-score标准化 (按基因)")

            # 合并数据以计算全局统计量
            combined = pd.concat([tumor_transformed, normal_transformed], axis=1)

            # 计算每个基因的均值和标准差
            gene_mean = combined.mean(axis=1)
            gene_std = combined.std(axis=1)

            # 标准化 (避免除零)
            tumor_transformed = (tumor_transformed.subtract(gene_mean, axis=0)).divide(
                gene_std.replace(0, 1), axis=0)
            normal_transformed = (normal_transformed.subtract(gene_mean, axis=0)).divide(
                gene_std.replace(0, 1), axis=0)

        elif normalize == 'minmax':
            if self.verbose:
                print(f"✓ Min-Max标准化 [0, 1]")

            combined = pd.concat([tumor_transformed, normal_transformed], axis=1)
            gene_min = combined.min(axis=1)
            gene_max = combined.max(axis=1)
            gene_range = gene_max - gene_min

            tumor_transformed = (tumor_transformed.subtract(gene_min, axis=0)).divide(
                gene_range.replace(0, 1), axis=0)
            normal_transformed = (normal_transformed.subtract(gene_min, axis=0)).divide(
                gene_range.replace(0, 1), axis=0)

        # 显示转换前后的统计信息
        if self.verbose and log_transform:
            print(f"\n转换后统计 (肿瘤样本):")
            print(f"  - 均值: {tumor_transformed.values.mean():.2f}")
            print(f"  - 标准差: {tumor_transformed.values.std():.2f}")
            print(f"  - 范围: [{tumor_transformed.values.min():.2f}, {tumor_transformed.values.max():.2f}]")

        return tumor_transformed, normal_transformed

    def save_preprocessed_data(self, tumor_data: pd.DataFrame,
                              normal_data: pd.DataFrame,
                              suffix: str = 'preprocessed') -> None:
        """
        保存预处理后的数据

        Parameters:
        -----------
        tumor_data : pd.DataFrame
            肿瘤样本数据
        normal_data : pd.DataFrame
            正常样本数据
        suffix : str
            文件名后缀
        """
        tumor_file = os.path.join(config.RESULTS_DIR, f'tumor_{suffix}.csv')
        normal_file = os.path.join(config.RESULTS_DIR, f'normal_{suffix}.csv')

        tumor_data.to_csv(tumor_file)
        normal_data.to_csv(normal_file)

        if self.verbose:
            print(f"\n✓ 保存预处理数据:")
            print(f"  - 肿瘤: {tumor_file}")
            print(f"  - 正常: {normal_file}")

    def run_preprocessing_pipeline(self, expression_file: str,
                                   phenotype_file: str = None,
                                   filter_lowexp: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        执行完整的预处理流程

        Parameters:
        -----------
        expression_file : str
            表达矩阵文件路径
        phenotype_file : str, optional
            样本注释文件路径
        filter_lowexp : bool
            是否过滤低表达基因

        Returns:
        --------
        tumor_data : pd.DataFrame
            预处理后的肿瘤样本数据
        normal_data : pd.DataFrame
            预处理后的正常样本数据
        combined_data : pd.DataFrame
            合并的数据 (用于后续分析)
        """
        # 1. 加载数据
        self.load_data(expression_file, phenotype_file)

        # 2. 分离样本
        tumor_data, normal_data = self.separate_samples()

        # 3. 过滤低表达基因
        if filter_lowexp:
            combined_for_filtering = pd.concat([tumor_data, normal_data], axis=1)
            combined_filtered = self.filter_low_expression(combined_for_filtering)

            # 应用过滤
            tumor_data = tumor_data.loc[combined_filtered.index]
            normal_data = normal_data.loc[combined_filtered.index]

        # 4. 表达值转换和标准化
        tumor_transformed, normal_transformed = self.transform_expression(
            tumor_data, normal_data
        )

        # 5. 合并数据
        combined_data = pd.concat([tumor_transformed, normal_transformed], axis=1)

        # 6. 保存结果
        if config.SAVE_INTERMEDIATE:
            self.save_preprocessed_data(tumor_transformed, normal_transformed)

        if self.verbose:
            print("\n" + "=" * 60)
            print("✓ 预处理完成!")
            print("=" * 60)
            print(f"\n最终数据规模:")
            print(f"  - 基因数: {combined_data.shape[0]:,}")
            print(f"  - 肿瘤样本数: {tumor_transformed.shape[1]}")
            print(f"  - 正常样本数: {normal_transformed.shape[1]}")
            print(f"  - 总样本数: {combined_data.shape[1]}")

        return tumor_transformed, normal_transformed, combined_data


# ============================================================================
# 便捷函数
# ============================================================================
def preprocess_tcga_data(expression_file: str,
                        phenotype_file: str = None,
                        filter_lowexp: bool = True,
                        verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    便捷函数: 执行TCGA数据预处理

    Parameters:
    -----------
    expression_file : str
        表达矩阵文件路径
    phenotype_file : str, optional
        样本注释文件路径
    filter_lowexp : bool
        是否过滤低表达基因
    verbose : bool
        是否打印详细信息

    Returns:
    --------
    tumor_data : pd.DataFrame
        预处理后的肿瘤样本数据
    normal_data : pd.DataFrame
        预处理后的正常样本数据
    combined_data : pd.DataFrame
        合并的数据
    """
    preprocessor = TCGAPreprocessor(verbose=verbose)
    return preprocessor.run_preprocessing_pipeline(
        expression_file, phenotype_file, filter_lowexp
    )


if __name__ == "__main__":
    # 测试代码
    print("模块1: TCGA数据预处理")
    print("=" * 60)
    print("\n本模块包含以下主要功能:")
    print("1. load_data() - 加载表达矩阵和样本信息")
    print("2. separate_samples() - 识别肿瘤/正常样本")
    print("3. filter_low_expression() - 过滤低表达基因")
    print("4. transform_expression() - 表达值转换和标准化")
    print("5. run_preprocessing_pipeline() - 完整流程")
    print("\n请通过main_pipeline.py调用此模块")
