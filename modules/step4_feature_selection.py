"""
模块4: 特征选择 (Feature Selection)

功能:
1. 使用Lasso回归筛选关键lncRNA生物标志物
2. 使用Random Forest评估特征重要性
3. 选择top N个最具预测性的lncRNA

方法学原理:
- Lasso (L1正则化): 将不重要特征的系数压缩为0,自动进行特征选择
- Random Forest: 通过Gini不纯度下降评估特征重要性
- 为什么使用Lasso?
  * 高维数据: 基因表达数据特征数>>样本数
  * 稀疏性: 真正的生物标志物数量有限
  * 共线性: lncRNA之间可能存在共表达关系
  * 可解释性: 最终模型简洁,易于解释

作者: Bioinformatics Ph.D. Student
日期: 2025
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import os
import sys
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FeatureSelector:
    """
    特征选择器

    主要功能:
    - Lasso回归特征选择
    - Random Forest特征重要性评估
    - 综合多种方法选择最优特征子集
    """

    def __init__(self, verbose=True):
        """
        初始化特征选择器

        Parameters:
        -----------
        verbose : bool
            是否打印详细信息
        """
        self.verbose = verbose
        self.selected_features = []
        self.feature_scores = {}
        self.lasso_model = None
        self.rf_model = None

    def prepare_data(self, tumor_data: pd.DataFrame,
                    normal_data: pd.DataFrame,
                    de_genes: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备特征选择的数据

        Parameters:
        -----------
        tumor_data : pd.DataFrame
            肿瘤样本表达矩阵
        normal_data : pd.DataFrame
            正常样本表达矩阵
        de_genes : list, optional
            差异表达基因列表 (如果提供,仅使用这些基因)

        Returns:
        --------
        X : np.ndarray
            特征矩阵 (samples x features)
        y : np.ndarray
            标签向量 (0=normal, 1=tumor)
        feature_names : list
            特征名称列表
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("特征选择: 数据准备")
            print("=" * 60)

        # 合并数据
        combined = pd.concat([tumor_data, normal_data], axis=1)

        # 如果指定了DE基因,仅保留这些基因
        if de_genes:
            de_genes_present = [g for g in de_genes if g in combined.index]
            combined = combined.loc[de_genes_present]

            if self.verbose:
                print(f"\n使用差异表达基因: {len(de_genes_present)}")
        else:
            if self.verbose:
                print(f"\n使用全部基因")

        # 转置为样本x特征格式
        X = combined.T.values
        feature_names = combined.index.tolist()

        # 创建标签
        n_tumor = tumor_data.shape[1]
        n_normal = normal_data.shape[1]

        y = np.array([1] * n_tumor + [0] * n_normal)

        if self.verbose:
            print(f"✓ 数据准备完成:")
            print(f"  - 样本数: {X.shape[0]} (肿瘤:{n_tumor}, 正常:{n_normal})")
            print(f"  - 特征数: {X.shape[1]}")
            print(f"  - 类别分布: 肿瘤={n_tumor}, 正常={n_normal}")

        return X, y, feature_names

    def lasso_feature_selection(self, X: np.ndarray, y: np.ndarray,
                               feature_names: List[str],
                               alpha_range: List[float] = None,
                               cv_folds: int = None) -> Tuple[List[str], pd.DataFrame]:
        """
        使用Lasso回归进行特征选择

        Lasso原理:
        - 目标函数: min||y - Xw||^2 + α||w||_1
        - L1正则化使部分特征的系数为0
        - 通过交叉验证选择最优α值

        为什么选择Lasso:
        1. 稀疏解: 自动选择少数重要特征
        2. 可解释性: 清晰展示哪些lncRNA重要
        3. 共线性处理: 从相关特征中选择一个
        4. 泛化能力: 正则化防止过拟合

        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            标签向量
        feature_names : list
            特征名称
        alpha_range : list
            正则化参数搜索范围
        cv_folds : int
            交叉验证折数

        Returns:
        --------
        selected_features : list
            选择的特征列表
        coef_df : pd.DataFrame
            特征系数表
        """
        if alpha_range is None:
            alpha_range = config.LASSO_ALPHA_RANGE
        if cv_folds is None:
            cv_folds = config.LASSO_CV_FOLDS

        if self.verbose:
            print("\n" + "=" * 60)
            print("Lasso特征选择")
            print("=" * 60)
            print(f"\nLasso原理: L1正则化 -> 稀疏解 -> 自动特征选择")
            print(f"参数设置:")
            print(f"  - α搜索范围: {alpha_range}")
            print(f"  - 交叉验证: {cv_folds}折")

        # 标准化特征 (Lasso对特征尺度敏感)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Lasso回归 (使用交叉验证选择最优α)
        lasso_cv = LassoCV(
            alphas=alpha_range,
            cv=cv_folds,
            max_iter=config.LASSO_MAX_ITER,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )

        lasso_cv.fit(X_scaled, y)
        self.lasso_model = lasso_cv

        # 获取系数
        coefficients = lasso_cv.coef_

        # 创建特征系数表
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })

        # 选择非零系数的特征
        nonzero_df = coef_df[coef_df['coefficient'] != 0].copy()
        nonzero_df = nonzero_df.sort_values('abs_coefficient', ascending=False)

        selected_features = nonzero_df['feature'].tolist()

        if self.verbose:
            print(f"\n✓ Lasso训练完成:")
            print(f"  - 最优α值: {lasso_cv.alpha_:.4f}")
            print(f"  - 原始特征数: {len(feature_names)}")
            print(f"  - 非零系数特征: {len(selected_features)}")
            print(f"  - 稀疏度: {(1 - len(selected_features)/len(feature_names))*100:.1f}%")

            if len(selected_features) > 0:
                print(f"\nTop 10 Lasso选择的lncRNA:")
                for idx, row in nonzero_df.head(10).iterrows():
                    direction = "↑" if row['coefficient'] > 0 else "↓"
                    print(f"  {direction} {row['feature']}: |系数|={row['abs_coefficient']:.4f}")

        self.feature_scores['lasso'] = coef_df
        return selected_features, coef_df

    def randomforest_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                       feature_names: List[str],
                                       n_estimators: int = None,
                                       max_features: str = None) -> Tuple[List[str], pd.DataFrame]:
        """
        使用Random Forest评估特征重要性

        RF特征重要性原理:
        - 基于Gini不纯度下降
        - 或基于permutation importance
        - 集成学习,结果稳健

        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            标签向量
        feature_names : list
            特征名称
        n_estimators : int
            树的数量
        max_features : str
            每棵树考虑的最大特征数

        Returns:
        --------
        top_features : list
            按重要性排序的特征列表
        importance_df : pd.DataFrame
            特征重要性表
        """
        if n_estimators is None:
            n_estimators = config.RF_N_ESTIMATORS
        if max_features is None:
            max_features = config.RF_MAX_FEATURES

        if self.verbose:
            print("\n" + "=" * 60)
            print("Random Forest特征重要性评估")
            print("=" * 60)
            print(f"\n参数设置:")
            print(f"  - 树的数量: {n_estimators}")
            print(f"  - 最大特征数: {max_features}")

        # 训练Random Forest
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        )

        rf.fit(X, y)
        self.rf_model = rf

        # 获取特征重要性
        importances = rf.feature_importances_

        # 创建重要性表
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        importance_df = importance_df.sort_values('importance', ascending=False)

        # 所有特征按重要性排序
        top_features = importance_df['feature'].tolist()

        if self.verbose:
            print(f"\n✓ Random Forest训练完成:")
            print(f"  - OOB准确率: {rf.oob_score_:.3f}" if hasattr(rf, 'oob_score_') else "")
            print(f"  - 训练集准确率: {rf.score(X, y):.3f}")

            print(f"\nTop 10 重要lncRNA:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: 重要性={row['importance']:.4f}")

        self.feature_scores['randomforest'] = importance_df
        return top_features, importance_df

    def select_final_features(self, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str],
                             n_features: int = None,
                             method: str = None) -> List[str]:
        """
        选择最终的生物标志物集合

        策略:
        1. 如果method='lasso': 仅使用Lasso结果
        2. 如果method='rf': 仅使用RF结果
        3. 如果method='union': Lasso和RF的并集
        4. 如果method='intersection': Lasso和RF的交集
        5. 如果method='hybrid': Lasso为主,RF为辅排序

        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            标签向量
        feature_names : list
            特征名称
        n_features : int
            最终选择的特征数量
        method : str
            特征选择方法

        Returns:
        --------
        list : 最终选择的特征列表
        """
        if n_features is None:
            n_features = config.N_SELECTED_FEATURES
        if method is None:
            method = config.FEATURE_SELECTION_METHOD

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"选择最终{n_features}个生物标志物")
            print("=" * 60)
            print(f"\n方法: {method}")

        # 执行Lasso和RF
        lasso_features, lasso_df = self.lasso_feature_selection(X, y, feature_names)
        rf_features, rf_df = self.randomforest_feature_importance(X, y, feature_names)

        # 根据方法选择特征
        if method == 'lasso':
            # 仅使用Lasso选择的特征
            selected = lasso_features[:n_features]

        elif method == 'rf':
            # 仅使用RF top特征
            selected = rf_features[:n_features]

        elif method == 'union':
            # Lasso和RF的并集
            lasso_set = set(lasso_features[:n_features//2])
            rf_set = set(rf_features[:n_features//2])
            selected = list(lasso_set.union(rf_set))

        elif method == 'intersection':
            # Lasso和RF的交集
            lasso_set = set(lasso_features[:n_features*2])
            rf_set = set(rf_features[:n_features*2])
            selected = list(lasso_set.intersection(rf_set))

        elif method == 'hybrid':
            # 混合方法: Lasso非零特征 + RF重要性排序
            lasso_nonzero = set(lasso_features)
            rf_df_subset = rf_df[rf_df['feature'].isin(lasso_nonzero)]
            selected = rf_df_subset.nlargest(n_features, 'importance')['feature'].tolist()

        else:
            raise ValueError(f"未知的特征选择方法: {method}")

        # 确保不超过n_features
        selected = selected[:n_features]

        if self.verbose:
            print(f"\n✓ 特征选择完成:")
            print(f"  - 选择特征数: {len(selected)}")

            # 显示选择的特征及其在两种方法中的排名
            print(f"\n最终选择的生物标志物:")
            for i, feat in enumerate(selected, 1):
                lasso_rank = lasso_df[lasso_df['feature']==feat].index[0] if feat in lasso_df['feature'].values else "N/A"
                rf_rank = rf_df[rf_df['feature']==feat].index[0] if feat in rf_df['feature'].values else "N/A"

                print(f"  {i}. {feat}")
                print(f"     Lasso排名: #{lasso_rank+1 if isinstance(lasso_rank, int) else lasso_rank}, "
                      f"RF排名: #{rf_rank+1 if isinstance(rf_rank, int) else rf_rank}")

        self.selected_features = selected
        return selected

    def save_selected_features(self, output_file: str = None) -> None:
        """
        保存选择的特征及其评分

        Parameters:
        -----------
        output_file : str, optional
            输出文件路径
        """
        if output_file is None:
            output_file = os.path.join(
                config.RESULTS_DIR,
                config.OUTPUT_FILES['biomarkers']
            )

        if not self.selected_features:
            raise ValueError("尚未执行特征选择,请先调用select_final_features()")

        # 创建特征信息表
        feature_info = []

        for feat in self.selected_features:
            info = {
                'feature': feat,
                'selection_order': self.selected_features.index(feat) + 1
            }

            # 添加Lasso系数
            if 'lasso' in self.feature_scores:
                lasso_row = self.feature_scores['lasso'][
                    self.feature_scores['lasso']['feature'] == feat
                ]
                if not lasso_row.empty:
                    info['lasso_coefficient'] = lasso_row['coefficient'].values[0]
                    info['lasso_abs_coefficient'] = lasso_row['abs_coefficient'].values[0]

            # 添加RF重要性
            if 'randomforest' in self.feature_scores:
                rf_row = self.feature_scores['randomforest'][
                    self.feature_scores['randomforest']['feature'] == feat
                ]
                if not rf_row.empty:
                    info['rf_importance'] = rf_row['importance'].values[0]

            feature_info.append(info)

        result_df = pd.DataFrame(feature_info)
        result_df.to_csv(output_file, index=False)

        if self.verbose:
            print(f"\n✓ 保存特征选择结果:")
            print(f"  文件: {output_file}")
            print(f"  特征数: {len(result_df)}")

    def run_feature_selection_pipeline(self, tumor_data: pd.DataFrame,
                                       normal_data: pd.DataFrame,
                                       de_genes: List[str] = None,
                                       n_features: int = None,
                                       method: str = None) -> Tuple[List[str], pd.DataFrame]:
        """
        执行完整的特征选择流程

        Parameters:
        -----------
        tumor_data : pd.DataFrame
            肿瘤样本表达矩阵
        normal_data : pd.DataFrame
            正常样本表达矩阵
        de_genes : list, optional
            差异表达基因列表
        n_features : int
            最终选择的特征数量
        method : str
            特征选择方法

        Returns:
        --------
        selected_features : list
            选择的特征列表
        feature_matrix : pd.DataFrame
            仅包含选择特征的表达矩阵
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("特征选择流程")
            print("=" * 60)

        # 1. 准备数据
        X, y, feature_names = self.prepare_data(tumor_data, normal_data, de_genes)

        # 2. 执行特征选择
        selected = self.select_final_features(
            X, y, feature_names,
            n_features=n_features,
            method=method
        )

        # 3. 提取特征矩阵
        selected_present = [f for f in selected if f in tumor_data.index]
        feature_matrix = pd.concat([
            tumor_data.loc[selected_present],
            normal_data.loc[selected_present]
        ], axis=1)

        # 4. 保存结果
        if config.SAVE_INTERMEDIATE:
            self.save_selected_features()

        if self.verbose:
            print("\n" + "=" * 60)
            print("✓ 特征选择完成!")
            print("=" * 60)
            print(f"\n最终生物标志物矩阵:")
            print(f"  - lncRNA标志物数: {len(selected_present)}")
            print(f"  - 样本数: {feature_matrix.shape[1]}")

        return selected, feature_matrix


# ============================================================================
# 便捷函数
# ============================================================================
def select_biomarkers(tumor_data: pd.DataFrame,
                     normal_data: pd.DataFrame,
                     de_genes: List[str] = None,
                     n_features: int = 20,
                     method: str = 'lasso',
                     verbose: bool = True) -> Tuple[List[str], pd.DataFrame]:
    """
    便捷函数: 执行特征选择

    Parameters:
    -----------
    tumor_data : pd.DataFrame
        肿瘤样本表达矩阵
    normal_data : pd.DataFrame
        正常样本表达矩阵
    de_genes : list, optional
        差异表达基因列表
    n_features : int
        选择的特征数量
    method : str
        特征选择方法
    verbose : bool
        是否打印详细信息

    Returns:
    --------
    list : 选择的特征列表
    pd.DataFrame : 特征矩阵
    """
    selector = FeatureSelector(verbose=verbose)
    return selector.run_feature_selection_pipeline(
        tumor_data, normal_data, de_genes, n_features, method
    )


if __name__ == "__main__":
    # 测试代码
    print("模块4: 特征选择")
    print("=" * 60)
    print("\n本模块包含以下主要功能:")
    print("1. prepare_data() - 数据准备")
    print("2. lasso_feature_selection() - Lasso特征选择")
    print("3. randomforest_feature_importance() - RF特征重要性")
    print("4. select_final_features() - 综合特征选择")
    print("5. run_feature_selection_pipeline() - 完整流程")
    print("\n请通过main_pipeline.py调用此模块")
