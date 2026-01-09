"""
模块5: 机器学习分类 (Machine Learning Classification)

功能:
1. 构建SVM/Random Forest分类器
2. 训练lncRNA生物标志物预测模型
3. 模型评估和性能分析
4. 交叉验证和ROC分析

分类任务:
- 二分类: 肿瘤 (1) vs 正常 (0)
- 输入: 选择的lncRNA生物标志物表达值
- 输出: 肿瘤概率预测

模型选择:
- SVM: 适合小样本高维数据,核函数处理非线性
- RF: 提供概率估计,对异常值鲁棒

评估指标:
- AUC-ROC: 区分能力
- Accuracy: 整体准确率
- Sensitivity: 灵敏度(真阳性率)
- Specificity: 特异度(真阴性率)

作者: Bioinformatics Ph.D. Student
日期: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import os
import sys
from typing import Tuple, Dict, List
import pickle
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BiomarkerClassifier:
    """
    基于lncRNA生物标志物的分类器

    主要功能:
    - 训练SVM/RF分类模型
    - 评估模型性能
    - 生成预测结果
    """

    def __init__(self, classifier_type: str = None, verbose=True):
        """
        初始化分类器

        Parameters:
        -----------
        classifier_type : str
            分类器类型 ('svm' 或 'rf')
        verbose : bool
            是否打印详细信息
        """
        if classifier_type is None:
            classifier_type = config.CLASSIFIER

        self.classifier_type = classifier_type
        self.verbose = verbose
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.evaluation_results = {}

    def prepare_training_data(self, tumor_data: pd.DataFrame,
                             normal_data: pd.DataFrame,
                             test_size: float = None,
                             stratify: bool = None) -> None:
        """
        准备训练和测试数据

        Parameters:
        -----------
        tumor_data : pd.DataFrame
            肿瘤样本表达矩阵 (features x samples)
        normal_data : pd.DataFrame
            正常样本表达矩阵
        test_size : float
            测试集比例
        stratify : bool
            是否分层采样
        """
        if test_size is None:
            test_size = config.TEST_SIZE
        if stratify is None:
            stratify = config.STRATIFY

        if self.verbose:
            print("\n" + "=" * 60)
            print("准备训练数据")
            print("=" * 60)

        # 转置为样本x特征格式
        X_tumor = tumor_data.T
        X_normal = normal_data.T

        # 创建标签
        y_tumor = np.ones(X_tumor.shape[0])
        y_normal = np.zeros(X_normal.shape[0])

        # 合并数据
        X = pd.concat([X_tumor, X_normal]).values
        y = np.concatenate([y_tumor, y_normal])

        # 划分训练集和测试集
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=config.RANDOM_STATE,
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=config.RANDOM_STATE
            )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        if self.verbose:
            print(f"\n✓ 数据划分完成:")
            print(f"  - 训练集: {len(X_train)} 样本 (肿瘤:{sum(y_train)}, 正常:{len(y_train)-sum(y_train)})")
            print(f"  - 测试集: {len(X_test)} 样本 (肿瘤:{sum(y_test)}, 正常:{len(y_test)-sum(y_test)})")
            print(f"  - 特征数: {X_train.shape[1]}")

    def train_svm(self, C: float = None, kernel: str = None,
                  gamma: str = None, probability: bool = None) -> None:
        """
        训练SVM分类器

        SVM原理:
        - 寻找最优超平面最大化类别间隔
        - 核技巧处理非线性可分数据
        - 参数C控制正则化强度

        为什么选择SVM:
        1. 高维数据: 基因表达数据特征数>>样本数,SVM表现优异
        2. 小样本: 即使样本量较小也能获得良好泛化
        3. 核函数: RBF核可以捕获复杂的非线性关系
        4. 稳健性: 对过拟合有较好的抵抗

        Parameters:
        -----------
        C : float
            正则化参数 (越大越不regularization)
        kernel : str
            核函数类型
        gamma : str
            RBF核参数
        probability : bool
            是否输出概率估计
        """
        if C is None:
            C = config.SVM_C
        if kernel is None:
            kernel = config.SVM_KERNEL
        if gamma is None:
            gamma = config.SVM_GAMMA
        if probability is None:
            probability = config.SVM_PROBABILITY

        if self.verbose:
            print("\n" + "=" * 60)
            print("训练SVM分类器")
            print("=" * 60)
            print(f"\nSVM参数设置:")
            print(f"  - C (正则化): {C}")
            print(f"  - 核函数: {kernel}")
            print(f"  - Gamma: {gamma}")
            print(f"  - 概率估计: {probability}")

        # 标准化特征 (SVM对特征尺度敏感)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        # 训练SVM
        svm_model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            class_weight=config.SVM_CLASS_WEIGHT,
            random_state=config.RANDOM_STATE
        )

        svm_model.fit(X_train_scaled, self.y_train)
        self.model = svm_model

        # 预测
        self.y_pred = svm_model.predict(X_test_scaled)
        if probability:
            self.y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

        if self.verbose:
            print(f"\n✓ SVM训练完成:")
            print(f"  - 支持向量数: {svm_model.n_support_.sum()}")
            print(f"  - 训练集准确率: {svm_model.score(X_train_scaled, self.y_train):.3f}")
            print(f"  - 测试集准确率: {svm_model.score(X_test_scaled, self.y_test):.3f}")

    def train_random_forest(self, n_estimators: int = None,
                           max_depth: int = None) -> None:
        """
        训练Random Forest分类器

        RF原理:
        - 集成多棵决策树
        - 通过投票或平均做出预测
        - 降低单一决策树的方差

        RF优势:
        1. 不需要特征标准化
        2. 处理非线性关系
        3. 提供特征重要性
        4. 对异常值和缺失值鲁棒

        Parameters:
        -----------
        n_estimators : int
            树的数量
        max_depth : int
            树的最大深度
        """
        if n_estimators is None:
            n_estimators = config.RF_CLF_N_ESTIMATORS
        if max_depth is None:
            max_depth = config.RF_CLF_MAX_DEPTH

        if self.verbose:
            print("\n" + "=" * 60)
            print("训练Random Forest分类器")
            print("=" * 60)
            print(f"\nRF参数设置:")
            print(f"  - 树的数量: {n_estimators}")
            print(f"  - 最大深度: {max_depth}")

        # 训练RF
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=config.RF_CLF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.RF_CLF_MIN_SAMPLES_LEAF,
            class_weight='balanced',
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )

        rf_model.fit(self.X_train, self.y_train)
        self.model = rf_model

        # 预测
        self.y_pred = rf_model.predict(self.X_test)
        self.y_pred_proba = rf_model.predict_proba(self.X_test)[:, 1]

        if self.verbose:
            print(f"\n✓ RF训练完成:")
            print(f"  - 训练集准确率: {rf_model.score(self.X_train, self.y_train):.3f}")
            print(f"  - 测试集准确率: {rf_model.score(self.X_test, self.y_test):.3f}")

    def evaluate_model(self) -> Dict[str, float]:
        """
        评估模型性能

        计算多种评估指标:
        - Accuracy: 整体准确率
        - Precision: 精确率 (预测为肿瘤中真正是肿瘤的比例)
        - Recall (Sensitivity): 召回率 (真正肿瘤被识别出的比例)
        - F1-score: Precision和Recall的调和平均
        - AUC-ROC: 曲线下面积

        Returns:
        --------
        dict : 评估指标字典
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("模型性能评估")
            print("=" * 60)

        # 基本指标
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, zero_division=0)

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 计算灵敏度和特异度
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # AUC-ROC
        if self.y_pred_proba is not None:
            fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
            roc_auc = auc(fpr, tpr)

            self.evaluation_results['roc_curve'] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc
            }
        else:
            roc_auc = None

        # 汇总结果
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': roc_auc,
            'confusion_matrix': cm,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }

        self.evaluation_results.update(results)

        if self.verbose:
            print(f"\n性能指标:")
            print(f"  ✓ 准确率 (Accuracy): {accuracy:.3f}")
            print(f"  ✓ 精确率 (Precision): {precision:.3f}")
            print(f"  ✓ 召回率/灵敏度 (Recall/Sensitivity): {recall:.3f}")
            print(f"  ✓ 特异度 (Specificity): {specificity:.3f}")
            print(f"  ✓ F1分数: {f1:.3f}")

            if roc_auc is not None:
                print(f"  ✓ AUC-ROC: {roc_auc:.3f}")

            print(f"\n混淆矩阵:")
            print(f"                预测正常    预测肿瘤")
            print(f"  实际正常       {tn:3d}        {fp:3d}")
            print(f"  实际肿瘤       {fn:3d}        {tp:3d}")

        return results

    def cross_validate(self, n_folds: int = None) -> Dict[str, float]:
        """
        执行交叉验证

        交叉验证目的:
        - 更可靠的性能估计
        - 减少单次划分的偶然性
        - 评估模型稳定性

        Parameters:
        -----------
        n_folds : int
            交叉验证折数

        Returns:
        --------
        dict : CV结果
        """
        if n_folds is None:
            n_folds = config.CV_FOLDS

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"{n_folds}折交叉验证")
            print("=" * 60)

        # 合并训练和测试数据
        X_all = np.vstack([self.X_train, self.X_test])
        y_all = np.concatenate([self.y_train, self.y_test])

        # 标准化(如果使用SVM)
        if self.scaler is not None:
            X_all = self.scaler.fit_transform(X_all)

        # 创建模型
        if self.classifier_type == 'svm':
            model = SVC(
                C=config.SVM_C,
                kernel=config.SVM_KERNEL,
                gamma=config.SVM_GAMMA,
                class_weight=config.SVM_CLASS_WEIGHT,
                random_state=config.RANDOM_STATE
            )
        else:  # rf
            model = RandomForestClassifier(
                n_estimators=config.RF_CLF_N_ESTIMATORS,
                max_depth=config.RF_CLF_MAX_DEPTH,
                class_weight='balanced',
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )

        # 执行交叉验证
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)

        cv_scores = cross_val_score(
            model, X_all, y_all,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )

        if self.verbose:
            print(f"\n✓ 交叉验证完成:")
            print(f"  - 各折准确率: {[f'{s:.3f}' for s in cv_scores]}")
            print(f"  - 平均准确率: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

    def save_model(self, output_file: str = None) -> None:
        """
        保存训练好的模型

        Parameters:
        -----------
        output_file : str, optional
            模型保存路径
        """
        if output_file is None:
            output_file = os.path.join(
                config.RESULTS_DIR,
                config.OUTPUT_FILES['model']
            )

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'classifier_type': self.classifier_type,
            'evaluation_results': self.evaluation_results
        }

        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)

        if self.verbose:
            print(f"\n✓ 模型已保存: {output_file}")

    def save_predictions(self, output_file: str = None) -> None:
        """
        保存预测结果

        Parameters:
        -----------
        output_file : str, optional
            输出文件路径
        """
        if output_file is None:
            output_file = os.path.join(
                config.RESULTS_DIR,
                config.OUTPUT_FILES['predictions']
            )

        results_df = pd.DataFrame({
            'true_label': self.y_test,
            'predicted_label': self.y_pred,
            'probability': self.y_pred_proba if self.y_pred_proba is not None else np.nan
        })

        results_df.to_csv(output_file, index=False)

        if self.verbose:
            print(f"✓ 预测结果已保存: {output_file}")

    def save_classification_report(self, output_file: str = None) -> None:
        """
        保存详细的分类报告

        Parameters:
        -----------
        output_file : str, optional
            输出文件路径
        """
        if output_file is None:
            output_file = os.path.join(
                config.RESULTS_DIR,
                config.OUTPUT_FILES['classification_report']
            )

        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("lncRNA生物标志物分类模型报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"分类器类型: {self.classifier_type.upper()}\n")
            f.write(f"任务: 肿瘤 vs 正常二分类\n\n")

            f.write("性能指标:\n")
            f.write("-" * 40 + "\n")
            for key, value in self.evaluation_results.items():
                if key not in ['roc_curve', 'confusion_matrix']:
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")

            f.write("\n混淆矩阵:\n")
            f.write("-" * 40 + "\n")
            cm = self.evaluation_results['confusion_matrix']
            f.write(f"                预测正常    预测肿瘤\n")
            f.write(f"  实际正常       {cm[0,0]:3d}        {cm[0,1]:3d}\n")
            f.write(f"  实际肿瘤       {cm[1,0]:3d}        {cm[1,1]:3d}\n\n")

            f.write("分类详细报告:\n")
            f.write("-" * 40 + "\n")
            report = classification_report(
                self.y_test, self.y_pred,
                target_names=['Normal', 'Tumor'],
                zero_division=0
            )
            f.write(report)

        if self.verbose:
            print(f"✓ 分类报告已保存: {output_file}")

    def run_classification_pipeline(self, tumor_data: pd.DataFrame,
                                    normal_data: pd.DataFrame) -> Dict[str, any]:
        """
        执行完整的分类流程

        Parameters:
        -----------
        tumor_data : pd.DataFrame
            肿瘤样本表达矩阵
        normal_data : pd.DataFrame
            正常样本表达矩阵

        Returns:
        --------
        dict : 所有评估结果
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"{self.classifier_type.upper()}分类流程")
            print("=" * 60)

        # 1. 准备数据
        self.prepare_training_data(tumor_data, normal_data)

        # 2. 训练模型
        if self.classifier_type == 'svm':
            self.train_svm()
        elif self.classifier_type == 'rf':
            self.train_random_forest()
        else:
            raise ValueError(f"未知的分类器类型: {self.classifier_type}")

        # 3. 评估模型
        results = self.evaluate_model()

        # 4. 交叉验证
        cv_results = self.cross_validate()

        # 5. 保存结果
        if config.SAVE_INTERMEDIATE:
            self.save_model()
            self.save_predictions()
            self.save_classification_report()

        if self.verbose:
            print("\n" + "=" * 60)
            print("✓ 分类流程完成!")
            print("=" * 60)

        return {**results, **cv_results}


# ============================================================================
# 便捷函数
# ============================================================================
def train_classifier(tumor_data: pd.DataFrame,
                    normal_data: pd.DataFrame,
                    classifier_type: str = 'svm',
                    verbose: bool = True) -> Dict[str, any]:
    """
    便捷函数: 训练分类器

    Parameters:
    -----------
    tumor_data : pd.DataFrame
        肿瘤样本表达矩阵
    normal_data : pd.DataFrame
        正常样本表达矩阵
    classifier_type : str
        分类器类型 ('svm' 或 'rf')
    verbose : bool
        是否打印详细信息

    Returns:
    --------
    dict : 评估结果
    """
    classifier = BiomarkerClassifier(classifier_type, verbose)
    return classifier.run_classification_pipeline(tumor_data, normal_data)


if __name__ == "__main__":
    # 测试代码
    print("模块5: 机器学习分类")
    print("=" * 60)
    print("\n本模块包含以下主要功能:")
    print("1. prepare_training_data() - 准备训练数据")
    print("2. train_svm() - 训练SVM分类器")
    print("3. train_random_forest() - 训练RF分类器")
    print("4. evaluate_model() - 评估模型性能")
    print("5. run_classification_pipeline() - 完整流程")
    print("\n请通过main_pipeline.py调用此模块")
