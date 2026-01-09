"""
模块6: 可视化 (Visualization)

功能:
1. 火山图 (Volcano Plot) - 展示差异表达分析结果
2. 热图 (Heatmap) - 展示top生物标志物的表达模式
3. ROC曲线 (ROC Curve) - 展示分类器性能

可视化原则:
- 学术发表质量 (高DPI, 清晰标注)
- 色盲友好的配色方案
- 信息密集但易于理解
- 符合期刊要求

作者: Bioinformatics Ph.D. Student
日期: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import sys
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class BiomarkerVisualizer:
    """
    生物标志物分析可视化器

    主要功能:
    - 火山图: 差异表达格局
    - 热图: 生物标志物表达模式
    - ROC曲线: 分类器性能
    """

    def __init__(self, verbose=True):
        """
        初始化可视化器

        Parameters:
        -----------
        verbose : bool
            是否打印详细信息
        """
        self.verbose = verbose
        self.figures = {}

    def plot_volcano(self, de_results: pd.DataFrame,
                    log2fc_threshold: float = None,
                    fdr_threshold: float = None,
                    top_n: int = None,
                    output_file: str = None,
                    dpi: int = None) -> plt.Figure:
        """
        绘制火山图 (Volcano Plot)

        火山图解读:
        - X轴: Log2 Fold Change (表达差异)
          - 正值: 肿瘤中上调
          - 负值: 肿瘤中下调
        - Y轴: -log10(FDR) (统计显著性)
        - 显著基因: 右上(显著上调) 和左下(显著下调)

        参数意义:
        - Log2FC阈值: 生物学显著性(如>1表示2倍差异)
        - FDR阈值: 统计学显著性(如<0.05)

        Parameters:
        -----------
        de_results : pd.DataFrame
            差异表达结果表 (必须包含log2FC和qvalue列)
        log2fc_threshold : float
            Log2FC阈值
        fdr_threshold : float
            FDR阈值
        top_n : int
            标注top N个基因名称
        output_file : str, optional
            输出文件路径
        dpi : int
            图像分辨率

        Returns:
        --------
        plt.Figure : 火山图对象
        """
        if log2fc_threshold is None:
            log2fc_threshold = config.VOLCANO_LOG2FC
        if fdr_threshold is None:
            fdr_threshold = config.VOLCANO_FDR
        if top_n is None:
            top_n = config.VOLCANO_TOP_N
        if dpi is None:
            dpi = config.FIGURE_DPI

        if self.verbose:
            print("\n" + "=" * 60)
            print("绘制火山图")
            print("=" * 60)

        # 准备数据
        de_results_copy = de_results.copy()
        de_results_copy['neg_log10_fdr'] = -np.log10(de_results_copy['qvalue'] + 1e-300)

        # 标记显著基因
        de_results_copy['significance'] = 'Not Significant'

        upreg_mask = (de_results_copy['log2FC'] >= log2fc_threshold) & \
                    (de_results_copy['qvalue'] < fdr_threshold)
        downreg_mask = (de_results_copy['log2FC'] <= -log2fc_threshold) & \
                      (de_results_copy['qvalue'] < fdr_threshold)

        de_results_copy.loc[upreg_mask, 'significance'] = 'Up-regulated'
        de_results_copy.loc[downreg_mask, 'significance'] = 'Down-regulated'

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))

        # 定义颜色
        colors = {
            'Not Significant': '#CCCCCC',
            'Up-regulated': '#D62728',  # 红色
            'Down-regulated': '#1F77B4'  # 蓝色
        }

        # 绘制散点
        for category, color in colors.items():
            data = de_results_copy[de_results_copy['significance'] == category]
            ax.scatter(
                data['log2FC'],
                data['neg_log10_fdr'],
                c=color,
                label=category,
                alpha=0.6,
                s=20,
                edgecolors='none'
            )

        # 添加阈值线
        ax.axhline(y=-np.log10(fdr_threshold), color='black', linestyle='--',
                  linewidth=1, alpha=0.5, label=f'FDR = {fdr_threshold}')
        ax.axvline(x=log2fc_threshold, color='black', linestyle='--',
                  linewidth=1, alpha=0.5, label=f'Log2FC = ±{log2fc_threshold}')
        ax.axvline(x=-log2fc_threshold, color='black', linestyle='--',
                  linewidth=1, alpha=0.5)

        # 标注top基因
        top_up = de_results_copy[de_results_copy['significance'] == 'Up-regulated'].head(top_n//2)
        top_down = de_results_copy[de_results_copy['significance'] == 'Down-regulated'].head(top_n//2)

        for idx, row in pd.concat([top_up, top_down]).iterrows():
            ax.annotate(
                row['gene_id'][:20],  # 限制基因名长度
                xy=(row['log2FC'], row['neg_log10_fdr']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )

        # 设置标签和标题
        ax.set_xlabel('Log2 Fold Change (Tumor vs Normal)', fontsize=12, fontweight='bold')
        ax.set_ylabel('-Log10(FDR)', fontsize=12, fontweight='bold')
        ax.set_title('Differential Expression Analysis - lncRNA Biomarkers',
                    fontsize=14, fontweight='bold', pad=20)

        # 图例
        ax.legend(loc='upper right', frameon=True, shadow=True)

        # 网格
        ax.grid(True, alpha=0.3)

        # 统计信息
        n_up = upreg_mask.sum()
        n_down = downreg_mask.sum()

        textstr = f'Total: {len(de_results_copy)} genes\n'
        textstr += f'Up-regulated: {n_up}\n'
        textstr += f'Down-regulated: {n_down}'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

        plt.tight_layout()

        # 保存
        if output_file is None:
            output_file = os.path.join(config.FIGURES_DIR, 'volcano_plot.png')

        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')

        if self.verbose:
            print(f"\n✓ 火山图已生成:")
            print(f"  - 上调lncRNA: {n_up}")
            print(f"  - 下调lncRNA: {n_down}")
            print(f"  - 保存: {output_file}")

        self.figures['volcano'] = fig
        return fig

    def plot_heatmap(self, expression_data: pd.DataFrame,
                    tumor_samples: List[str],
                    normal_samples: List[str],
                    n_biomarkers: int = None,
                    standard_scale: int = None,
                    cluster_samples: bool = None,
                    cluster_genes: bool = None,
                    output_file: str = None,
                    dpi: int = None) -> plt.Figure:
        """
        绘制热图 (Heatmap)

        热图解读:
        - 行: lncRNA生物标志物
        - 列: 样本
        - 颜色: 表达水平 (红色=高表达, 蓝色=低表达)
        - 聚类: 层次聚类识别表达模式相似的样本/基因

        用途:
        - 可视化top生物标志物的表达模式
        - 识别样本亚型
        - 展示biomarker的区分能力

        Parameters:
        -----------
        expression_data : pd.DataFrame
            表达矩阵 (genes x samples)
        tumor_samples : list
            肿瘤样本ID列表
        normal_samples : list
            正常样本ID列表
        n_biomarkers : int
            显示的biomarker数量
        standard_scale : int
            标准化方式 (0=按行, 1=按列, None=不标准化)
        cluster_samples : bool
            是否对样本聚类
        cluster_genes : bool
            是否对基因聚类
        output_file : str, optional
            输出文件路径
        dpi : int
            图像分辨率

        Returns:
        --------
        plt.Figure : 热图对象
        """
        if n_biomarkers is None:
            n_biomarkers = config.HEATMAP_N_BIOMARKERS
        if standard_scale is None:
            standard_scale = config.HEATMAP_STANDARD_SCALE
        if cluster_samples is None:
            cluster_samples = config.HEATMAP_CLUSTER_SAMPLES
        if cluster_genes is None:
            cluster_genes = config.HEATMAP_CLUSTER_GENES
        if dpi is None:
            dpi = config.FIGURE_DPI

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"绘制热图 (Top {n_biomarkers} 生物标志物)")
            print("=" * 60)

        # 选择top N个基因 (基于方差)
        gene_var = expression_data.var(axis=1)
        top_genes = gene_var.nlargest(n_biomarkers).index.tolist()

        # 提取数据
        plot_data = expression_data.loc[top_genes]

        # 创建样本类型标签
        sample_types = []
        for sample in plot_data.columns:
            if sample in tumor_samples:
                sample_types.append('Tumor')
            elif sample in normal_samples:
                sample_types.append('Normal')
            else:
                sample_types.append('Unknown')

        # 绘制热图
        fig, ax = plt.subplots(figsize=(max(12, len(sample_types) * 0.3),
                                       max(8, n_biomarkers * 0.2)))

        # 定义颜色映射
        cmap = sns.diverging_palette(240, 10, as_cmap=True)

        # 绘制聚类热图
        g = sns.clustermap(
            plot_data,
            cmap=cmap,
            center=0,
            standard_scale=standard_scale,
            row_cluster=cluster_genes,
            col_cluster=cluster_samples,
            figsize=(max(12, len(sample_types) * 0.3), max(8, n_biomarkers * 0.2)),
            cbar_kws={'label': 'Expression Level (z-score)'},
            xticklabels=False,
            yticklabels=True,
            dendrogram_ratio=0.1 if not cluster_genes else 0.15
        )

        # 添加样本类型颜色条
        col_colors = ['#D62728' if t == 'Tumor' else '#1F77B4' for t in sample_types]
        g.ax_col_dendrogram.bar(0, 1, 1, 1, color=col_colors, transform=g.ax_col_dendrogram.transData)

        # 设置标题
        g.fig.suptitle(f'Top {n_biomarkers} lncRNA Biomarkers Expression Heatmap',
                      fontsize=14, fontweight='bold', y=0.98)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#D62728', label='Tumor'),
            Patch(facecolor='#1F77B4', label='Normal')
        ]
        g.ax_heatmap.legend(
            handles=legend_elements,
            loc='upper right',
            bbox_to_anchor=(1.15, 1.0),
            frameon=True,
            shadow=True
        )

        plt.tight_layout()

        # 保存
        if output_file is None:
            output_file = os.path.join(config.FIGURES_DIR, 'heatmap_top20.png')

        g.fig.savefig(output_file, dpi=dpi, bbox_inches='tight')

        if self.verbose:
            print(f"\n✓ 热图已生成:")
            print(f"  - 基因数: {n_biomarkers}")
            tumor_count = sample_types.count('Tumor')
            normal_count = sample_types.count('Normal')
            print(f"  - 样本数: {len(sample_types)} (肿瘤:{tumor_count}, 正常:{normal_count})")
            print(f"  - 保存: {output_file}")

        self.figures['heatmap'] = g.fig
        return g.fig

    def plot_roc_curve(self, evaluation_results: Dict,
                      cv_folds: int = None,
                      show_ci: bool = None,
                      output_file: str = None,
                      dpi: int = None) -> plt.Figure:
        """
        绘制ROC曲线 (Receiver Operating Characteristic)

        ROC曲线解读:
        - X轴: False Positive Rate (1-Specificity)
        - Y轴: True Positive Rate (Sensitivity)
        - 对角线: 随机猜测 (AUC=0.5)
        - 左上角: 完美分类 (AUC=1.0)

        AUC含义:
        - 0.9-1.0: 优秀
        - 0.8-0.9: 良好
        - 0.7-0.8: 一般
        - 0.5-0.7: 较差
        - 0.5: 无区分能力

        Parameters:
        -----------
        evaluation_results : dict
            评估结果字典 (必须包含roc_curve)
        cv_folds : int
            交叉验证折数
        show_ci : bool
            是否显示置信区间
        output_file : str, optional
            输出文件路径
        dpi : int
            图像分辨率

        Returns:
        --------
        plt.Figure : ROC曲线图
        """
        if cv_folds is None:
            cv_folds = config.ROC_CV_FOLDS
        if show_ci is None:
            show_ci = config.ROC_SHOW_CI
        if dpi is None:
            dpi = config.FIGURE_DPI

        if self.verbose:
            print("\n" + "=" * 60)
            print("绘制ROC曲线")
            print("=" * 60)

        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 8))

        if 'roc_curve' in evaluation_results:
            roc_data = evaluation_results['roc_curve']
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            roc_auc = roc_data['auc']

            # 绘制ROC曲线
            ax.plot(fpr, tpr, color='#D62728', linewidth=2,
                   label=f'ROC Curve (AUC = {roc_auc:.3f})')

            # 填充曲线下区域
            ax.fill_between(fpr, tpr, alpha=0.3, color='#D62728')

        # 绘制对角线(随机猜测)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--',
               linewidth=1, label='Random Guess (AUC = 0.500)')

        # 设置坐标轴
        ax.set_xlabel('False Positive Rate (1 - Specificity)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)',
                     fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve - lncRNA Biomarker Classification',
                    fontsize=14, fontweight='bold', pad=20)

        # 设置范围
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        # 图例
        ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)

        # 网格
        ax.grid(True, alpha=0.3)

        # 添加性能指标文本
        if 'accuracy' in evaluation_results:
            metrics_text = f'Performance Metrics:\n'
            metrics_text += f"Accuracy: {evaluation_results['accuracy']:.3f}\n"
            metrics_text += f"Sensitivity: {evaluation_results['sensitivity']:.3f}\n"
            metrics_text += f"Specificity: {evaluation_results['specificity']:.3f}\n"

            if 'auc' in evaluation_results and evaluation_results['auc']:
                metrics_text += f"AUC: {evaluation_results['auc']:.3f}"

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.55, 0.35, metrics_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', bbox=props)

        plt.tight_layout()

        # 保存
        if output_file is None:
            output_file = os.path.join(config.FIGURES_DIR, 'roc_curve.png')

        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')

        if self.verbose:
            print(f"\n✓ ROC曲线已生成:")
            if 'auc' in evaluation_results and evaluation_results['auc']:
                print(f"  - AUC: {evaluation_results['auc']:.3f}")
            print(f"  - 保存: {output_file}")

        self.figures['roc'] = fig
        return fig

    def create_summary_figure(self, de_results: pd.DataFrame,
                             expression_data: pd.DataFrame,
                             tumor_samples: List[str],
                             normal_samples: List[str],
                             evaluation_results: Dict,
                             output_file: str = None,
                             dpi: int = None) -> plt.Figure:
        """
        创建综合总结图

        包含: 火山图 + 热图 + ROC曲线

        Parameters:
        -----------
        de_results : pd.DataFrame
            差异表达结果
        expression_data : pd.DataFrame
            表达数据
        tumor_samples : list
            肿瘤样本
        normal_samples : list
            正常样本
        evaluation_results : dict
            评估结果
        output_file : str, optional
            输出文件
        dpi : int
            分辨率

        Returns:
        --------
        plt.Figure : 综合图
        """
        if dpi is None:
            dpi = config.FIGURE_DPI

        if self.verbose:
            print("\n" + "=" * 60)
            print("生成综合总结图")
            print("=" * 60)

        # 创建3面板图形
        fig = plt.figure(figsize=(20, 6))

        # 子图1: 火山图
        ax1 = plt.subplot(1, 3, 1)

        de_results_copy = de_results.copy()
        de_results_copy['neg_log10_fdr'] = -np.log10(de_results_copy['qvalue'] + 1e-300)

        colors = ['#CCCCCC', '#D62728', '#1F77B4']
        categories = ['Not Significant', 'Up-regulated', 'Down-regulated']

        for cat, col in zip(categories, colors):
            if cat == 'Up-regulated':
                mask = (de_results_copy['log2FC'] >= config.VOLCANO_LOG2FC) & \
                       (de_results_copy['qvalue'] < config.VOLCANO_FDR)
            elif cat == 'Down-regulated':
                mask = (de_results_copy['log2FC'] <= -config.VOLCANO_LOG2FC) & \
                       (de_results_copy['qvalue'] < config.VOLCANO_FDR)
            else:
                mask = ~((de_results_copy['log2FC'] >= config.VOLCANO_LOG2FC) & \
                        (de_results_copy['qvalue'] < config.VOLCANO_FDR)) & \
                       ~((de_results_copy['log2FC'] <= -config.VOLCANO_LOG2FC) & \
                        (de_results_copy['qvalue'] < config.VOLCANO_FDR))

            ax1.scatter(de_results_copy[mask]['log2FC'],
                       de_results_copy[mask]['neg_log10_fdr'],
                       c=col, label=cat, alpha=0.6, s=15)

        ax1.axhline(y=-np.log10(config.VOLCANO_FDR), color='black',
                   linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.axvline(x=config.VOLCANO_LOG2FC, color='black',
                   linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.axvline(x=-config.VOLCANO_LOG2FC, color='black',
                   linestyle='--', linewidth=0.8, alpha=0.5)

        ax1.set_xlabel('Log2FC', fontsize=10, fontweight='bold')
        ax1.set_ylabel('-Log10(FDR)', fontsize=10, fontweight='bold')
        ax1.set_title('A. Differential Expression', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 子图2: 热图 (简化版)
        ax2 = plt.subplot(1, 3, 2)

        # 选择top 20基因
        gene_var = expression_data.var(axis=1)
        top_genes = gene_var.nlargest(20).index.tolist()
        plot_data = expression_data.loc[top_genes]

        # 简单热图(无聚类)
        sns.heatmap(plot_data, cmap='coolwarm', center=0,
                   cbar_kws={'label': 'Z-score'},
                   xticklabels=False, yticklabels=True, ax=ax2)

        ax2.set_title('B. Biomarker Expression', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Samples', fontsize=10, fontweight='bold')
        ax2.set_ylabel('lncRNA Biomarkers', fontsize=10, fontweight='bold')

        # 子图3: ROC曲线
        ax3 = plt.subplot(1, 3, 3)

        if 'roc_curve' in evaluation_results:
            fpr = evaluation_results['roc_curve']['fpr']
            tpr = evaluation_results['roc_curve']['tpr']
            auc_score = evaluation_results['roc_curve']['auc']

            ax3.plot(fpr, tpr, color='#D62728', linewidth=2,
                    label=f'AUC = {auc_score:.3f}')
            ax3.fill_between(fpr, tpr, alpha=0.3, color='#D62728')

        ax3.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
        ax3.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold')
        ax3.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold')
        ax3.set_title('C. Model Performance', fontsize=12, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=10)
        ax3.grid(True, alpha=0.3)

        plt.suptitle('lncRNA Biomarker Discovery for TCGA-LIHC',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # 保存
        if output_file is None:
            output_file = os.path.join(config.FIGURES_DIR, 'summary_figure.png')

        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')

        if self.verbose:
            print(f"\n✓ 综合总结图已生成: {output_file}")

        self.figures['summary'] = fig
        return fig


# ============================================================================
# 便捷函数
# ============================================================================
def create_all_visualizations(de_results: pd.DataFrame,
                             expression_data: pd.DataFrame,
                             tumor_samples: List[str],
                             normal_samples: List[str],
                             evaluation_results: Dict,
                             verbose: bool = True) -> Dict[str, plt.Figure]:
    """
    便捷函数: 创建所有可视化

    Parameters:
    -----------
    de_results : pd.DataFrame
        差异表达结果
    expression_data : pd.DataFrame
        表达数据
    tumor_samples : list
        肿瘤样本
    normal_samples : list
        正常样本
    evaluation_results : dict
        评估结果
    verbose : bool
        是否打印详细信息

    Returns:
    --------
    dict : 所有图形对象
    """
    visualizer = BiomarkerVisualizer(verbose=verbose)

    # 创建各图形
    visualizer.plot_volcano(de_results)
    visualizer.plot_heatmap(expression_data, tumor_samples, normal_samples)
    visualizer.plot_roc_curve(evaluation_results)
    visualizer.create_summary_figure(
        de_results, expression_data,
        tumor_samples, normal_samples,
        evaluation_results
    )

    return visualizer.figures


if __name__ == "__main__":
    # 测试代码
    print("模块6: 可视化")
    print("=" * 60)
    print("\n本模块包含以下主要功能:")
    print("1. plot_volcano() - 绘制火山图")
    print("2. plot_heatmap() - 绘制热图")
    print("3. plot_roc_curve() - 绘制ROC曲线")
    print("4. create_summary_figure() - 创建综合总结图")
    print("\n请通过main_pipeline.py调用此模块")
