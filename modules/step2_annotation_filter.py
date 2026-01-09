"""
模块2: lncRNA注释过滤

功能:
1. 识别lncRNA基因 (基于Ensembl基因ID规则)
2. 从表达矩阵中过滤出仅包含lncRNA的数据
3. 生成lncRNA注释信息

生物学背景:
- Ensembl基因ID格式: ENSG[编号]
- 蛋白编码基因通常: ENSG00000001 ~ ENSG00060000
- lncRNA基因通常: ENSG00060001 ~ ENSG00100000+
- 基于GENCODE数据库的基因分类

作者: Bioinformatics Ph.D. Student
日期: 2025
"""

import pandas as pd
import numpy as np
import os
import sys
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LncRNAAnnotator:
    """
    lncRNA注释过滤器

    主要功能:
    - 从Ensembl基因ID识别lncRNA
    - 过滤表达矩阵保留lncRNA
    - 生成lncRNA注释信息
    """

    def __init__(self, verbose=True):
        """
        初始化注释器

        Parameters:
        -----------
        verbose : bool
            是否打印详细信息
        """
        self.verbose = verbose
        self.lncRNA_ids = []
        self.mrna_ids = []
        self.annotation_info = {}

    def parse_ensembl_id(self, gene_id: str) -> Dict[str, any]:
        """
        解析Ensembl基因ID

        Ensembl ID格式: ENSG[X]<number>
        - ENSG: Ensembl基因标识符
        - <number>: 唯一编号

        基于GENCODE v41的基因分布:
        - Protein-coding: ~20,000个基因 (ID < 60000)
        - lncRNA: ~50,000个基因 (ID > 60000)
        - 其他非编码RNA (miRNA, snRNA等): 散布

        Parameters:
        -----------
        gene_id : str
            Ensembl基因ID (如 ENSG00000123456)

        Returns:
        --------
        dict : 包含ID解析信息的字典
        """
        # 提取数字部分
        match = re.search(r'ENSG\d+(\.\d+)?', gene_id)

        if not match:
            return {
                'is_ensembl': False,
                'gene_type': 'Unknown',
                'id_number': None
            }

        # 获取数字部分 (去除版本号 .1, .2等)
        id_str = gene_id.split('.')[0]
        id_number = int(id_str.replace('ENSG', ''))

        # 判断基因类型 (基于ID编号范围)
        if id_number < config.ENSEMBL_LNCRNA_ID_START:
            gene_type = 'protein_coding'
        elif id_number <= config.ENSEMBL_LNCRNA_ID_END:
            gene_type = 'lncRNA'
        else:
            # 超高ID编号可能是其他非编码RNA或假基因
            gene_type = 'other_noncoding'

        return {
            'is_ensembl': True,
            'gene_type': gene_type,
            'id_number': id_number
        }

    def identify_lncrnas(self, gene_ids: List[str],
                        use_known_list: bool = False,
                        known_lncrna_file: str = None) -> List[str]:
        """
        识别lncRNA基因ID

        方法1: 基于Ensembl ID编号规则 (默认)
        方法2: 使用已知lncRNA列表 (如果提供)

        Parameters:
        -----------
        gene_ids : list
            基因ID列表
        use_known_list : bool
            是否使用已知lncRNA列表
        known_lncrna_file : str, optional
            已知lncRNA列表文件路径

        Returns:
        --------
        list : lncRNA基因ID列表
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("步骤: lncRNA注释过滤")
            print("=" * 60)

        # 方法1: 使用已知列表
        if use_known_list and known_lncrna_file and os.path.exists(known_lncrna_file):
            if self.verbose:
                print("\n方法: 使用已知lncRNA列表")

            known_lncrnas = pd.read_csv(known_lncrna_file, header=None)[0].tolist()
            known_lncrnas_set = set(known_lncrnas)

            self.lncRNA_ids = [gid for gid in gene_ids if gid in known_lncrnas_set]
            self.mrna_ids = [gid for gid in gene_ids if gid not in known_lncrnas_set]

            if self.verbose:
                print(f"✓ 从文件加载 {len(known_lncrnas):,} 个已知lncRNA")

        # 方法2: 基于Ensembl ID规则 (默认)
        else:
            if self.verbose:
                print("\n方法: 基于Ensembl基因ID编号规则")
                print(f"  - lncRNA阈值: ID > {config.ENSEMBL_LNCRNA_ID_START:,}")

            lncRNA_list = []
            mrna_list = []
            other_list = []

            for gene_id in gene_ids:
                parsed = self.parse_ensembl_id(gene_id)

                if parsed['is_ensembl']:
                    if parsed['gene_type'] == 'lncRNA':
                        lncRNA_list.append(gene_id)
                    elif parsed['gene_type'] == 'protein_coding':
                        mrna_list.append(gene_id)
                    else:
                        other_list.append(gene_id)
                else:
                    # 非Ensembl ID (如Gene Symbol)
                    other_list.append(gene_id)

            self.lncRNA_ids = lncRNA_list
            self.mrna_ids = mrna_list

            if self.verbose:
                print(f"\n✓ 基因分类完成:")
                print(f"  - 总基因数: {len(gene_ids):,}")
                print(f"  - lncRNA: {len(lncRNA_list):,} ({len(lncRNA_list)/len(gene_ids)*100:.1f}%)")
                print(f"  - mRNA (蛋白编码): {len(mrna_list):,} ({len(mrna_list)/len(gene_ids)*100:.1f}%)")
                print(f"  - 其他: {len(other_list):,} ({len(other_list)/len(gene_ids)*100:.1f}%)")

        return self.lncRNA_ids

    def filter_expression_matrix(self, expression_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        过滤表达矩阵,分离lncRNA和mRNA

        Parameters:
        -----------
        expression_data : pd.DataFrame
            完整表达矩阵 (genes x samples)

        Returns:
        --------
        lncRNA_matrix : pd.DataFrame
            仅包含lncRNA的表达矩阵
        mrna_matrix : pd.DataFrame
            仅包含mRNA的表达矩阵
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("步骤: 过滤表达矩阵")

        if not self.lncRNA_ids:
            # 如果还没识别lncRNA,先执行识别
            all_gene_ids = expression_data.index.tolist()
            self.identify_lncrnas(all_gene_ids)

        # 过滤lncRNA
        lncRNA_present = [gid for gid in self.lncRNA_ids if gid in expression_data.index]
        lncRNA_matrix = expression_data.loc[lncRNA_present]

        # 过滤mRNA
        mrna_present = [gid for gid in self.mrna_ids if gid in expression_data.index]
        mrna_matrix = expression_data.loc[mrna_present]

        if self.verbose:
            print(f"\n✓ 过滤完成:")
            print(f"  - 原始基因数: {expression_data.shape[0]:,}")
            print(f"  - lncRNA矩阵: {lncRNA_matrix.shape[0]:,} 基因 × {lncRNA_matrix.shape[1]} 样本")
            print(f"  - mRNA矩阵: {mrna_matrix.shape[0]:,} 基因 × {mrna_matrix.shape[1]} 样本")

        return lncRNA_matrix, mrna_matrix

    def generate_annotation_info(self) -> pd.DataFrame:
        """
        生成lncRNA注释信息

        Returns:
        --------
        pd.DataFrame : 包含lncRNA注释的DataFrame
        """
        annotation_data = []

        for lncRNA_id in self.lncRNA_ids:
            parsed = self.parse_ensembl_id(lncRNA_id)

            annotation_data.append({
                'gene_id': lncRNA_id,
                'id_number': parsed['id_number'],
                'gene_type': 'lncRNA',
                'source': 'Ensembl'
            })

        self.annotation_info = pd.DataFrame(annotation_data)

        if self.verbose:
            print(f"\n✓ 生成注释信息: {len(self.annotation_info)} 个lncRNA")

        return self.annotation_info

    def save_filtered_data(self, lncRNA_matrix: pd.DataFrame,
                          save_annotation: bool = True) -> None:
        """
        保存过滤后的数据

        Parameters:
        -----------
        lncRNA_matrix : pd.DataFrame
            lncRNA表达矩阵
        save_annotation : bool
            是否保存注释信息
        """
        # 保存lncRNA表达矩阵
        output_file = os.path.join(config.RESULTS_DIR, 'lncRNA_expression_matrix.csv')
        lncRNA_matrix.to_csv(output_file)

        if self.verbose:
            print(f"\n✓ 保存lncRNA表达矩阵:")
            print(f"  文件: {output_file}")

        # 保存注释信息
        if save_annotation:
            annotation_df = self.generate_annotation_info()
            annotation_file = os.path.join(config.RESULTS_DIR, 'lncRNA_annotation.csv')
            annotation_df.to_csv(annotation_file, index=False)

            if self.verbose:
                print(f"  注释: {annotation_file}")

    def run_annotation_pipeline(self, expression_data: pd.DataFrame,
                                save_results: bool = True) -> pd.DataFrame:
        """
        执行完整的注释过滤流程

        Parameters:
        -----------
        expression_data : pd.DataFrame
            原始表达矩阵
        save_results : bool
            是否保存结果

        Returns:
        --------
        lncRNA_matrix : pd.DataFrame
            仅包含lncRNA的表达矩阵
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("lncRNA注释过滤流程")
            print("=" * 60)

        # 1. 识别lncRNA
        all_gene_ids = expression_data.index.tolist()
        self.identify_lncrnas(all_gene_ids)

        # 2. 过滤表达矩阵
        lncRNA_matrix, mrna_matrix = self.filter_expression_matrix(expression_data)

        # 3. 保存结果
        if save_results and config.SAVE_INTERMEDIATE:
            self.save_filtered_data(lncRNA_matrix)

        if self.verbose:
            print("\n" + "=" * 60)
            print("✓ lncRNA注释过滤完成!")
            print("=" * 60)
            print(f"\n最终lncRNA数据规模:")
            print(f"  - lncRNA基因数: {lncRNA_matrix.shape[0]:,}")
            print(f"  - 样本数: {lncRNA_matrix.shape[1]}")
            print(f"\n这些lncRNA将用于后续的差异表达分析和生物标志物发现")

        return lncRNA_matrix


# ============================================================================
# 便捷函数
# ============================================================================
def filter_lncrnas(expression_data: pd.DataFrame,
                   verbose: bool = True) -> pd.DataFrame:
    """
    便捷函数: 从表达矩阵中过滤lncRNA

    Parameters:
    -----------
    expression_data : pd.DataFrame
        原始表达矩阵 (genes x samples)
    verbose : bool
        是否打印详细信息

    Returns:
    --------
    pd.DataFrame : 仅包含lncRNA的表达矩阵
    """
    annotator = LncRNAAnnotator(verbose=verbose)
    return annotator.run_annotation_pipeline(expression_data)


def generate_synthetic_lncRNA_ids(n_genes: int = 5000) -> List[str]:
    """
    生成模拟的lncRNA Ensembl ID列表

    用于创建合成数据时使用

    Parameters:
    -----------
    n_genes : int
        要生成的lncRNA数量

    Returns:
    --------
    list : lncRNA Ensembl ID列表
    """
    # 生成在lncRNA ID范围内的随机编号
    start_id = config.ENSEMBL_LNCRNA_ID_START
    end_id = config.ENSEMBL_LNCRNA_ID_END

    # 随机选择ID编号
    id_numbers = np.random.choice(
        range(start_id, end_id),
        size=min(n_genes, end_id - start_id),
        replace=False
    )

    # 转换为Ensembl ID格式
    lncRNA_ids = [f'ENSG{id_number:011d}' for id_number in sorted(id_numbers)]

    return lncRNA_ids


if __name__ == "__main__":
    # 测试代码
    print("模块2: lncRNA注释过滤")
    print("=" * 60)
    print("\n本模块包含以下主要功能:")
    print("1. parse_ensembl_id() - 解析Ensembl基因ID")
    print("2. identify_lncrnas() - 识别lncRNA基因")
    print("3. filter_expression_matrix() - 过滤表达矩阵")
    print("4. run_annotation_pipeline() - 完整流程")
    print("\n请通过main_pipeline.py调用此模块")
