"""
EEG特征分析和可视化模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 尝试导入seaborn，如果失败则使用matplotlib替代
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("警告: seaborn未安装，将使用matplotlib绘制相关性矩阵")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EEGFeatureAnalyzer:
    """EEG特征分析器"""
    
    def __init__(self, features_dir: str):
        self.features_dir = Path(features_dir)
        self.df = None
        
    def load_features(self, filename: str = "all_features_summary.csv"):
        """加载特征数据"""
        file_path = self.features_dir / filename
        if file_path.exists():
            self.df = pd.read_csv(file_path)
            print(f"成功加载 {len(self.df)} 条特征数据")
            return True
        else:
            print(f"特征文件不存在: {file_path}")
            return False
    
    def get_feature_columns(self) -> List[str]:
        """获取特征列名"""
        if self.df is None:
            return []
        
        exclude_cols = ['filename', 'subject_id', 'condition', 'discharge_point', 
                       'channel_name', 'sampling_rate']
        return [col for col in self.df.columns if col not in exclude_cols]
    
    def basic_statistics(self):
        """基本统计分析"""
        if self.df is None:
            print("请先加载特征数据")
            return
        
        print("=== EEG特征基本统计 ===")
        print(f"总样本数: {len(self.df)}")
        print(f"被试数量: {self.df['subject_id'].nunique()}")
        print(f"通道数量: {self.df['channel_name'].nunique()}")
        print(f"条件类型: {list(self.df['condition'].unique())}")
        
        print("\n各条件样本分布:")
        condition_counts = self.df['condition'].value_counts()
        for condition, count in condition_counts.items():
            print(f"  {condition}: {count}")
        
        print("\n被试放电点分布:")
        discharge_stats = self.df.groupby('subject_id')['discharge_point'].max()
        print(f"  平均放电点数: {discharge_stats.mean():.2f}")
        print(f"  最大放电点数: {discharge_stats.max()}")
        print(f"  最小放电点数: {discharge_stats.min()}")
    
    def feature_distribution_analysis(self):
        """特征分布分析"""
        if self.df is None:
            return
        
        feature_cols = self.get_feature_columns()
        numeric_features = [col for col in feature_cols if self.df[col].dtype in ['float64', 'int64']]
        
        # 主要功率特征分析
        power_features = ['total_power', 'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
        existing_power_features = [f for f in power_features if f in numeric_features]
        
        print("\n=== 功率特征统计 ===")
        for feature in existing_power_features:
            data = self.df[feature]
            print(f"{feature}:")
            print(f"  均值: {data.mean():.6f}")
            print(f"  标准差: {data.std():.6f}")
            print(f"  中位数: {data.median():.6f}")
            print(f"  范围: [{data.min():.6f}, {data.max():.6f}]")
    
    def compare_conditions(self):
        """比较放电前后条件"""
        if self.df is None:
            return
        
        feature_cols = self.get_feature_columns()
        numeric_features = [col for col in feature_cols if self.df[col].dtype in ['float64', 'int64']]
        
        # 按条件分组
        before_data = self.df[self.df['condition'] == 'Beforedis']
        after_data = self.df[self.df['condition'] == 'Afterdis']
        
        if len(before_data) == 0 or len(after_data) == 0:
            print("缺少放电前或放电后的数据")
            return
        
        print("\n=== 放电前后特征比较 ===")
        significant_features = []
        
        for feature in numeric_features[:10]:  # 只显示前10个特征
            before_mean = before_data[feature].mean()
            after_mean = after_data[feature].mean()
            
            # 简单的差异分析
            relative_change = ((after_mean - before_mean) / (before_mean + 1e-12)) * 100
            
            print(f"{feature}:")
            print(f"  放电前: {before_mean:.6f}")
            print(f"  放电后: {after_mean:.6f}")
            print(f"  相对变化: {relative_change:.2f}%")
            
            if abs(relative_change) > 10:  # 变化超过10%认为是显著的
                significant_features.append((feature, relative_change))
        
        if significant_features:
            print("\n显著变化的特征 (>10%):")
            for feature, change in significant_features:
                print(f"  {feature}: {change:.2f}%")
    
    def channel_analysis(self):
        """通道分析"""
        if self.df is None:
            return
        
        print("\n=== 通道分析 ===")
        channel_counts = self.df['channel_name'].value_counts()
        print(f"总通道数: {len(channel_counts)}")
        print("前10个通道的样本数:")
        for channel, count in channel_counts.head(10).items():
            print(f"  {channel}: {count}")
    
    def plot_power_distribution(self, save_dir: str = None):
        """绘制功率分布图"""
        if self.df is None:
            return
        
        power_features = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
        existing_features = [f for f in power_features if f in self.df.columns]
        
        if not existing_features:
            print("没有找到功率特征")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(existing_features):
            if i < len(axes):
                # 绘制直方图
                self.df[feature].hist(bins=50, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{feature} 分布')
                axes[i].set_xlabel('功率值')
                axes[i].set_ylabel('频数')
        
        # 隐藏多余的子图
        for i in range(len(existing_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'power_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"功率分布图保存至: {save_path}")
        
        plt.show()
    
    def plot_condition_comparison(self, features: List[str] = None, save_dir: str = None):
        """绘制条件比较图"""
        if self.df is None:
            return
        
        if features is None:
            features = ['alpha_power', 'beta_power', 'gamma_power', 'delta_power']
        
        existing_features = [f for f in features if f in self.df.columns]
        if not existing_features:
            print("没有找到指定的特征")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(existing_features[:4]):
            if i < len(axes):
                # 创建箱线图
                data_to_plot = []
                labels = []
                
                for condition in self.df['condition'].unique():
                    condition_data = self.df[self.df['condition'] == condition][feature]
                    data_to_plot.append(condition_data)
                    labels.append(condition)
                
                axes[i].boxplot(data_to_plot, labels=labels)
                axes[i].set_title(f'{feature} 条件比较')
                axes[i].set_ylabel('特征值')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'condition_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"条件比较图保存至: {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, features: List[str] = None, save_dir: str = None):
        """绘制特征相关性矩阵"""
        if self.df is None:
            return
        
        if features is None:
            # 选择主要特征
            main_features = ['total_power', 'delta_power', 'theta_power', 'alpha_power', 
                           'beta_power', 'gamma_power', 'SEF_90', 'SEF_95', 
                           'spectral_entropy', 'ADR', 'ATR', 'ABR']
            features = [f for f in main_features if f in self.df.columns]
        
        if len(features) < 2:
            print("特征数量不足以计算相关性")
            return
        
        # 计算相关性矩阵
        corr_matrix = self.df[features].corr()
        
        # 绘制热力图
        if HAS_SEABORN:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('EEG特征相关性矩阵')
        else:
            # 使用matplotlib替代方案
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # 设置刻度标签
            ax.set_xticks(range(len(features)))
            ax.set_yticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.set_yticklabels(features)
            
            # 添加数值标注
            for i in range(len(features)):
                for j in range(len(features)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black")
            
            # 添加颜色条
            plt.colorbar(im, shrink=0.8)
            plt.title('EEG特征相关性矩阵')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'correlation_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关性矩阵保存至: {save_path}")
        
        plt.show()
    
    def plot_subject_trends(self, subject_ids: List[str] = None, feature: str = 'alpha_power', 
                           save_dir: str = None):
        """绘制被试趋势图"""
        if self.df is None:
            return
        
        if feature not in self.df.columns:
            print(f"特征 {feature} 不存在")
            return
        
        if subject_ids is None:
            # 选择前5个被试
            subject_ids = list(self.df['subject_id'].unique())[:5]
        
        plt.figure(figsize=(12, 8))
        
        for subject_id in subject_ids:
            subject_data = self.df[self.df['subject_id'] == subject_id]
            
            if len(subject_data) == 0:
                continue
            
            # 按放电点和条件排序
            subject_data = subject_data.sort_values(['discharge_point', 'condition'])
            
            # 为每个条件创建不同的标记
            before_data = subject_data[subject_data['condition'] == 'Beforedis']
            after_data = subject_data[subject_data['condition'] == 'Afterdis']
            
            if len(before_data) > 0:
                plt.plot(before_data['discharge_point'], before_data[feature], 
                        'o-', label=f'{subject_id} (Before)', alpha=0.7)
            
            if len(after_data) > 0:
                plt.plot(after_data['discharge_point'], after_data[feature], 
                        's--', label=f'{subject_id} (After)', alpha=0.7)
        
        plt.xlabel('放电点')
        plt.ylabel(f'{feature}')
        plt.title(f'被试 {feature} 趋势图')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_dir:
            save_path = Path(save_dir) / f'subject_trends_{feature}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"被试趋势图保存至: {save_path}")
        
        plt.show()
    
    def generate_analysis_report(self, output_dir: str):
        """生成完整的分析报告"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("生成EEG特征分析报告...")
        
        # 基本统计
        self.basic_statistics()
        
        # 特征分布分析
        self.feature_distribution_analysis()
        
        # 条件比较
        self.compare_conditions()
        
        # 通道分析
        self.channel_analysis()
        
        # 生成图表
        print("\n生成可视化图表...")
        self.plot_power_distribution(str(output_path))
        self.plot_condition_comparison(save_dir=str(output_path))
        self.plot_correlation_matrix(save_dir=str(output_path))
        
        # 选择几个代表性被试绘制趋势图
        if len(self.df['subject_id'].unique()) > 0:
            sample_subjects = list(self.df['subject_id'].unique())[:3]
            self.plot_subject_trends(sample_subjects, 'alpha_power', str(output_path))
            self.plot_subject_trends(sample_subjects, 'spectral_entropy', str(output_path))
        
        print(f"\n分析报告生成完成，保存在: {output_path}")


def main():
    """主函数"""
    # 特征文件路径
    features_dir = "/mnt/d/project2025/BeforeTask_features(放电前后数据)"
    
    # 创建分析器
    analyzer = EEGFeatureAnalyzer(features_dir)
    
    # 加载特征数据
    if analyzer.load_features():
        # 生成完整分析报告
        analyzer.generate_analysis_report(features_dir)
    else:
        print("无法加载特征数据，请先运行特征提取脚本")


if __name__ == "__main__":
    main()
