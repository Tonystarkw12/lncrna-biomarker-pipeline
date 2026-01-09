"""
模块3: 差异表达分析 (Differential Expression Analysis)

功能:
1. 统计检验识别差异表达lncRNA
2. 计算Log2 Fold Change
3. 多重假设检验FDR校正
4. 生成差异表达结果表

统计学方法:
- Wilcoxon秩和检验 (非参数检验)
- t检验 (参数检验,假设正态分布)
- Benjamini-Hochberg FDR校正

生物学意义:
- 识别在肿瘤与正常样本中表达显著差异的lncRNA
- 这些lncRNA可能是潜在的诊断/治疗靶点

作者: Bioinformatics Ph.D. Student
日期: 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
import sys
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DifferentialExpressionAnalyzer:
    """
    差异表达分析器

    主要功能:
    - 执行统计检验(Wilcoxon/t-test)
    - 计算Fold Change
    - FDR多重假设检验校正
    - 识别显著差异表达的基因
    """

    def __init__(self, verbose=True):
        """
        初始化分析器

        Parameters:
        -----------
        verbose : bool
            是否打印详细信息
        """
        self.verbose = verbose
        self.de_results = None
        self.significant_genes = []

    def calculate_fold_change(self, tumor_data: pd.DataFrame,
                             normal_data: pd.DataFrame,
                             log2_data: bool = True) -> pd.Series:
        """
        计算Fold Change

        Fold Change = 肿瘤组平均表达 / 正常组平均表达
        Log2FC = log2(Fold Change)

        Parameters:
        -----------
        tumor_data : pd.DataFrame
            肿瘤样本表达矩阵
        normal_data : pd.DataFrame
            正常样本表达矩阵
        log2_data : bool
            输入数据是否已经是log2转换

        Returns:
        --------
        pd.Series : 每个基因的Log2FC
        """
        if self.verbose:
            print("\n计算Fold Change...")

        # 计算平均表达
        tumor_mean = tumor_data.mean(axis=1)
        normal_mean = normal_data.mean(axis=1)

        if log2_data:
            # 数据已经是log2尺度,直接相减
            log2fc = tumor_mean - normal_mean
        else:
            # 原始尺度,先log2再计算
            log2fc = np.log2(tumor_mean + 1) - np.log2(normal_mean + 1)

        if self.verbose:
            print(f"✓ Log2FC范围: [{log2fc.min():.2f}, {log2fc.max():.2f}]")
            print(f"  平均|Log2FC|: {np.abs(log2fc).mean():.2f}")

        return log2fc

    def perform_statistical_test(self, tumor_data: pd.DataFrame,
                                 normal_data: pd.DataFrame,
                                 method: str = None) -> Tuple[pd.Series, pd.Series]:
        """
        执行统计检验

        支持的方法:
        - 'wilcoxon': Wilcoxon秩和检验 (Mann-Whitney U test)
          - 非参数检验,不假设正态分布
          - 适合基因表达数据(通常非正态分布)
          - 推荐使用

        - 'ttest': Student's t检验
          - 参数检验,假设正态分布
          - 适合log2转换后近似正态的数据
          - 当样本量较大时可使用

        Parameters:
        -----------
        tumor_data : pd.DataFrame
            肿瘤样本表达矩阵
        normal_data : pd.DataFrame
            正常样本表达矩阵
        method : str
            统计检验方法 ('wilcoxon' 或 'ttest')

        Returns:
        --------
        pvalues : pd.Series
            原始p值
            statistics : pd.Series
            统计量 (U值或t值)
        """
        if method is None:
            method = config.DE_METHOD

        if self.verbose:
            print(f"\n执行统计检验: {method.upper()}")

        genes = tumor_data.index.tolist()
        pvalues = []
        statistics = []

        for gene in genes:
            tumor_expr = tumor_data.loc[gene].values
            normal_expr = normal_data.loc[gene].values

            # 移除NaN值
            tumor_expr = tumor_expr[~np.isnan(tumor_expr)]
            normal_expr = normal_expr[~np.isnan(normal_expr)]

            if method == 'wilcoxon':
                # Wilcoxon秩和检验
                try:
                    stat, pval = stats.mannwhitneyu(
                        tumor_expr, normal_expr,
                        alternative='two-sided'
                    )
                except:
                    # 如果计算失败,赋予p=1
                    stat, pval = 0, 1.0

            elif method == 'ttest':
                # Student's t检验
                try:
                    stat, pval = stats.ttest_ind(tumor_expr, normal_expr, equal_var=False)
                except:
                    stat, pval = 0, 1.0

            else:
                raise ValueError(f"未知的统计检验方法: {method}")

            pvalues.append(pval)
            statistics.append(stat)

        pvalues = pd.Series(pvalues, index=genes)
        statistics = pd.Series(statistics, index=genes)

        if self.verbose:
            print(f"✓ 完成统计检验: {len(genes)} 个基因")
            print(f"  p值范围: [{pvalues.min():.2e}, {pvalues.max():.2f}]")
            print(f"  p < 0.05: {(pvalues < 0.05).sum()} 个基因")
            print(f"  p < 0.01: {(pvalues < 0.01).sum()} 个基因")

        return pvalues, statistics

    def adjust_pvalues(self, pvalues: pd.Series,
                      method: str = None) -> pd.Series:
        """
        多重假设检验校正

        方法:
        - 'fdr_bh': Benjamini-Hochberg (推荐)
          - 控制False Discovery Rate
          - 适合高通量筛选

        - 'bonferroni': Bonferroni校正
          - 控制Family-wise Error Rate
          - 过于保守,可能导致假阴性

        Parameters:
        -----------
        pvalues : pd.Series
            原始p值
        method : str
            FDR校正方法

        Returns:
        --------
        pd.Series : 校正后的q值 (FDR)
        """
        if method is None:
            method = config.FDR_METHOD

        if self.verbose:
            print(f"\nFDR校正: {method}")

        # 执行多重检验校正
        rejected, qvalues, _, _ = multipletests(
            pvalues,
            alpha=0.05,
            method=method
        )

        qvalues = pd.Series(qvalues, index=pvalues.index)

        if self.verbose:
            print(f"✓ FDR校正完成:")
            print(f"  q值范围: [{qvalues.min():.2e}, {qvalues.max():.2e}]")
            print(f"  q < 0.05: {(qvalues < 0.05).sum()} 个基因")
            print(f"  q < 0.01: {(qvalues < 0.01).sum()} 个基因")

        return qvalues

    def identify_differential_expression(self, tumor_data: pd.DataFrame,
                                        normal_data: pd.DataFrame,
                                        log2fc_threshold: float = None,
                                        fdr_threshold: float = None,
                                        method: str = None) -> pd.DataFrame:
        """
        执行完整的差异表达分析

        流程:
        1. 计算Log2 Fold Change
        2. 执行统计检验
        3. FDR校正
        4. 筛选显著差异基因

        Parameters:
        -----------
        tumor_data : pd.DataFrame
            肿瘤样本表达矩阵
        normal_data : pd.DataFrame
            正常样本表达矩阵
        log2fc_threshold : float
            Log2FC阈值 (绝对值)
        fdr_threshold : float
            FDR阈值
        method : str
            统计检验方法

        Returns:
        --------
        pd.DataFrame : 差异表达结果表
        """
        if log2fc_threshold is None:
            log2fc_threshold = config.LOG2FC_THRESHOLD
        if fdr_threshold is None:
            fdr_threshold = config.FDR_THRESHOLD
        if method is None:
            method = config.DE_METHOD

        if self.verbose:
            print("\n" + "=" * 60)
            print("差异表达分析")
            print("=" * 60)
            print(f"\n参数设置:")
            print(f"  - 统计方法: {method}")
            print(f"  - Log2FC阈值: |{log2fc_threshold}|")
            print(f"  - FDR阈值: < {fdr_threshold}")

        # 1. 计算Log2FC
        log2fc = self.calculate_fold_change(tumor_data, normal_data, log2_data=True)

        # 2. 统计检验
        pvalues, statistics = self.perform_statistical_test(
            tumor_data, normal_data, method=method
        )

        # 3. FDR校正
        qvalues = self.adjust_pvalues(pvalues)

        # 4. 组装结果表
        results_df = pd.DataFrame({
            'gene_id': tumor_data.index,
            'log2FC': log2fc,
            'tumor_mean': tumor_data.mean(axis=1),
            'normal_mean': normal_data.mean(axis=1),
            'statistic': statistics,
            'pvalue': pvalues,
            'qvalue': qvalues,
            'significant': (np.abs(log2fc) >= log2fc_threshold) & (qvalues < fdr_threshold),
            'regulation': ['Up' if fc > 0 else 'Down' for fc in log2fc]
        })

        # 按q值排序
        results_df = results_df.sort_values('qvalue')

        # 提取显著基因
        significant_df = results_df[results_df['significant']].copy()
        self.significant_genes = significant_df['gene_id'].tolist()

        # 统计结果
        n_up = (significant_df['regulation'] == 'Up').sum()
        n_down = (significant_df['regulation'] == 'Down').sum()
        n_total = len(significant_df)

        if self.verbose:
            print("\n" + "=" * 60)
            print("✓ 差异表达分析完成!")
            print("=" * 60)
            print(f"\n显著差异表达lncRNA (|Log2FC|>={log2fc_threshold}, FDR<{fdr_threshold}):")
            print(f"  - 总数: {n_total}")
            print(f"  - 上调: {n_up}")
            print(f"  - 下调: {n_down}")
            print(f"  - 占比: {n_total/len(results_df)*100:.1f}%")

            if n_total > 0:
                print(f"\nTop 10 上调lncRNA:")
                top_up = significant_df[significant_df['regulation'] == 'Up'].head(10)
                for idx, row in top_up.iterrows():
                    print(f"  {row['gene_id']}: Log2FC={row['log2FC']:.2f}, FDR={row['qvalue']:.2e}")

                print(f"\nTop 10 下调lncRNA:")
                top_down = significant_df[significant_df['regulation'] == 'Down'].head(10)
                for idx, row in top_down.iterrows():
                    print(f"  {row['gene_id']}: Log2FC={row['log2FC']:.2f}, FDR={row['qvalue']:.2e}")

        self.de_results = results_df
        return results_df

    def save_results(self, output_file: str = None,
                    save_significant_only: bool = False) -> None:
        """
        保存差异表达分析结果

        Parameters:
        -----------
        output_file : str, optional
            输出文件路径
        save_significant_only : bool
            是否仅保存显著基因
        """
        if output_file is None:
            output_file = os.path.join(
                config.RESULTS_DIR,
                config.OUTPUT_FILES['differential_expression']
            )

        if self.de_results is None:
            raise ValueError("尚未执行差异表达分析,请先调用identify_differential_expression()")

        # 选择要保存的数据
        if save_significant_only:
            data_to_save = self.de_results[self.de_results['significant']]
        else:
            data_to_save = self.de_results

        data_to_save.to_csv(output_file, index=False)

        if self.verbose:
            print(f"\n✓ 保存差异表达结果:")
            print(f"  文件: {output_file}")
            print(f"  基因数: {len(data_to_save)}")

    def get_top_genes(self, n: int = 20,
                     regulation: str = 'both') -> pd.DataFrame:
        """
        获取Top差异表达基因

        Parameters:
        -----------
        n : int
            返回的基因数量
        regulation : str
            'up', 'down', 或 'both'

        Returns:
        --------
        pd.DataFrame : Top基因表
        """
        if self.de_results is None:
            raise ValueError("尚未执行差异表达分析")

        significant_df = self.de_results[self.de_results['significant']].copy()

        if regulation == 'up':
            top_df = significant_df[significant_df['regulation'] == 'Up'].head(n)
        elif regulation == 'down':
            top_df = significant_df[significant_df['regulation'] == 'Down'].head(n)
        else:  # 'both'
            top_df = significant_df.head(n)

        return top_df


# ============================================================================
# 便捷函数
# ============================================================================
def perform_de_analysis(tumor_data: pd.DataFrame,
                       normal_data: pd.DataFrame,
                       log2fc_threshold: float = 1.0,
                       fdr_threshold: float = 0.05,
                       method: str = 'wilcoxon',
                       verbose: bool = True) -> pd.DataFrame:
    """
    便捷函数: 执行差异表达分析

    Parameters:
    -----------
    tumor_data : pd.DataFrame
        肿瘤样本表达矩阵
    normal_data : pd.DataFrame
        正常样本表达矩阵
    log2fc_threshold : float
        Log2FC阈值
    fdr_threshold : float
        FDR阈值
    method : str
        统计检验方法
    verbose : bool
        是否打印详细信息

    Returns:
    --------
    pd.DataFrame : 差异表达结果表
    """
    analyzer = DifferentialExpressionAnalyzer(verbose=verbose)
    results = analyzer.identify_differential_expression(
        tumor_data, normal_data,
        log2fc_threshold=log2fc_threshold,
        fdr_threshold=fdr_threshold,
        method=method
    )

    # 自动保存结果
    if config.SAVE_INTERMEDIATE:
        analyzer.save_results()

    return results


if __name__ == "__main__":
    # 测试代码
    print("模块3: 差异表达分析")
    print("=" * 60)
    print("\n本模块包含以下主要功能:")
    print("1. calculate_fold_change() - 计算Log2 Fold Change")
    print("2. perform_statistical_test() - 执行统计检验")
    print("3. adjust_pvalues() - FDR多重检验校正")
    print("4. identify_differential_expression() - 完整分析流程")
    print("\n请通过main_pipeline.py调用此模块")
