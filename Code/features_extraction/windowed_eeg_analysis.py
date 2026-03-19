"""
EEG滑动窗口特征分析和可视化模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

class WindowedEEGAnalyzer:
    """滑动窗口EEG特征分析器"""
    
    def __init__(self, features_dir: str):
        self.features_dir = Path(features_dir)
        self.df = None
        self.feature_cols = None
        
    def load_windowed_features(self, filename: str = "all_windowed_features_summary.csv"):
        """加载滑动窗口特征数据"""
        file_path = self.features_dir / filename
        if file_path.exists():
            self.df = pd.read_csv(file_path)
            # 识别特征列
            id_cols = ['filename', 'subject_id', 'condition', 'trial_id', 'channel', 
                      'window_index', 'start_time', 'end_time']
            self.feature_cols = [col for col in self.df.columns if col not in id_cols]
            
            print(f"✅ 成功加载 {len(self.df)} 条窗口特征数据")
            print(f"📊 数据形状: {self.df.shape}")
            print(f"📝 特征数量: {len(self.feature_cols)}")
            print(f"👥 受试者数: {self.df['subject_id'].nunique()}")
            print(f"🧠 通道数: {self.df['channel'].nunique()}")
            print(f"🪟 窗口数范围: {self.df['window_index'].min()} - {self.df['window_index'].max()}")
            return True
        else:
            print(f"❌ 特征文件不存在: {file_path}")
            return False
    
    def load_single_file_features(self, filename: str):
        """加载单个文件的特征数据进行快速查看"""
        file_path = self.features_dir / filename
        if file_path.exists():
            self.df = pd.read_csv(file_path)
            # 识别特征列
            id_cols = ['filename', 'subject_id', 'condition', 'trial_id', 'channel', 
                      'window_index', 'start_time', 'end_time']
            self.feature_cols = [col for col in self.df.columns if col not in id_cols]
            
            print(f"✅ 成功加载单文件特征数据")
            print(f"📊 数据形状: {self.df.shape}")
            print(f"📝 特征数量: {len(self.feature_cols)}")
            print(f"🧠 通道数: {self.df['channel'].nunique()}")
            print(f"🪟 窗口数: {self.df['window_index'].nunique()}")
            return True
        else:
            print(f"❌ 文件不存在: {file_path}")
            return False
    
    def show_data_overview(self):
        """显示数据概览"""
        if self.df is None:
            print("❌ 请先加载数据")
            return
        
        print("\\n" + "="*60)
        print("📊 数据概览")
        print("="*60)
        
        # 基本信息
        print(f"数据形状: {self.df.shape}")
        print(f"内存使用: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # 分组统计
        print(f"\\n📝 分组统计:")
        print(f"  受试者数: {self.df['subject_id'].nunique()}")
        print(f"  条件类型: {self.df['condition'].unique()}")
        print(f"  通道数: {self.df['channel'].nunique()}")
        print(f"  窗口数: {self.df['window_index'].nunique()}")
        
        # 时间信息
        print(f"\\n⏰ 时间信息:")
        print(f"  时间范围: {self.df['start_time'].min():.1f} - {self.df['end_time'].max():.1f} 秒")
        print(f"  窗口间隔: {self.df['start_time'].diff().dropna().iloc[0]:.1f} 秒")
        
        # 特征统计
        print(f"\\n🎯 特征信息:")
        power_features = [col for col in self.feature_cols if 'power' in col]
        ratio_features = [col for col in self.feature_cols if 'ratio' in col]
        other_features = [col for col in self.feature_cols if col not in power_features + ratio_features]
        
        print(f"  功率特征: {len(power_features)} 个")
        print(f"  比例特征: {len(ratio_features)} 个") 
        print(f"  其他特征: {len(other_features)} 个")
        
        # 显示前几行
        print(f"\\n📋 数据样例:")
        display_cols = ['subject_id', 'condition', 'channel', 'window_index', 'start_time', 'end_time']
        display_cols.extend(power_features[:3])  # 显示前3个功率特征
        print(self.df[display_cols].head())
        
    def plot_time_series_features(self, channels: List[str] = None, features: List[str] = None, 
                                 subject_id: str = None, condition: str = None):
        """绘制时间序列特征图"""
        if self.df is None:
            print("❌ 请先加载数据")
            return
        
        # 筛选数据
        plot_df = self.df.copy()
        
        if subject_id:
            plot_df = plot_df[plot_df['subject_id'] == subject_id]
        if condition:
            plot_df = plot_df[plot_df['condition'] == condition]
        
        # 默认参数
        if channels is None:
            channels = plot_df['channel'].unique()[:4]  # 前4个通道
        if features is None:
            features = [col for col in self.feature_cols if 'power' in col][:5]  # 前5个功率特征
        
        # 创建子图
        fig, axes = plt.subplots(len(features), len(channels), 
                                figsize=(4*len(channels), 3*len(features)))
        fig.suptitle(f'时间序列特征 - {subject_id or "所有受试者"} - {condition or "所有条件"}', 
                    fontsize=16, y=0.98)
        
        if len(features) == 1:
            axes = [axes] if len(channels) == 1 else axes
        if len(channels) == 1:
            axes = [[ax] for ax in axes] if len(features) > 1 else [[axes]]
        
        for i, feature in enumerate(features):
            for j, channel in enumerate(channels):
                ax = axes[i][j] if len(features) > 1 else axes[j]
                
                # 筛选特定通道数据
                channel_data = plot_df[plot_df['channel'] == channel]
                
                if len(channel_data) > 0:
                    # 按时间排序
                    channel_data = channel_data.sort_values('start_time')
                    
                    # 绘制时间序列
                    ax.plot(channel_data['start_time'], channel_data[feature], 
                           marker='o', linewidth=2, markersize=4)
                    ax.set_title(f'{channel} - {feature}')
                    ax.set_xlabel('时间 (秒)')
                    ax.set_ylabel(feature)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{channel} - {feature}')
        
        plt.tight_layout()
        plt.show()
        
    def plot_channel_comparison(self, feature: str = 'total_power', window_index: int = 0):
        """比较不同通道在特定时间窗口的特征值"""
        if self.df is None:
            print("❌ 请先加载数据")
            return
        
        # 筛选特定窗口
        window_data = self.df[self.df['window_index'] == window_index]
        
        if len(window_data) == 0:
            print(f"❌ 窗口 {window_index} 无数据")
            return
        
        # 按通道分组统计
        channel_stats = window_data.groupby('channel')[feature].agg(['mean', 'std']).reset_index()
        channel_stats = channel_stats.sort_values('mean', ascending=False)
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 柱状图
        bars = ax1.bar(range(len(channel_stats)), channel_stats['mean'], 
                      yerr=channel_stats['std'], capsize=3, alpha=0.7)
        ax1.set_title(f'通道 {feature} 比较 (窗口 {window_index})')
        ax1.set_xlabel('通道')
        ax1.set_ylabel(feature)
        ax1.set_xticks(range(len(channel_stats)))
        ax1.set_xticklabels(channel_stats['channel'], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 热力图 (如果有多个受试者/条件)
        if window_data['subject_id'].nunique() > 1 or window_data['condition'].nunique() > 1:
            pivot_data = window_data.pivot_table(
                values=feature, 
                index='channel', 
                columns=['subject_id', 'condition'], 
                aggfunc='mean'
            )
            
            im = ax2.imshow(pivot_data.values, cmap='viridis', aspect='auto')
            ax2.set_title(f'{feature} 热力图 (窗口 {window_index})')
            ax2.set_xlabel('受试者-条件')
            ax2.set_ylabel('通道')
            ax2.set_yticks(range(len(pivot_data.index)))
            ax2.set_yticklabels(pivot_data.index)
            ax2.set_xticks(range(len(pivot_data.columns)))
            ax2.set_xticklabels([f'{s}-{c}' for s, c in pivot_data.columns], rotation=45)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax2)
        else:
            ax2.text(0.5, 0.5, '单一条件\\n无法生成热力图', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.show()
        
    def plot_feature_correlation(self, window_index: int = 0):
        """绘制特征相关性矩阵"""
        if self.df is None:
            print("❌ 请先加载数据")
            return
        
        # 筛选特定窗口的数值特征
        window_data = self.df[self.df['window_index'] == window_index]
        numeric_features = window_data[self.feature_cols].select_dtypes(include=[np.number])
        
        # 计算相关性
        corr_matrix = numeric_features.corr()
        
        # 绘制相关性热力图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title(f'特征相关性矩阵 (窗口 {window_index})', fontsize=14)
        plt.tight_layout()
        plt.show()
        
    def export_summary_stats(self, output_file: str = "windowed_features_summary_stats.csv"):
        """导出汇总统计"""
        if self.df is None:
            print("❌ 请先加载数据")
            return
        
        summary_stats = []
        
        for feature in self.feature_cols:
            feature_data = pd.to_numeric(self.df[feature], errors='coerce')
            
            stats = {
                'feature': feature,
                'count': feature_data.count(),
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'min': feature_data.min(),
                'q25': feature_data.quantile(0.25),
                'median': feature_data.median(),
                'q75': feature_data.quantile(0.75),
                'max': feature_data.max(),
                'skewness': feature_data.skew(),
                'kurtosis': feature_data.kurtosis()
            }
            summary_stats.append(stats)
        
        stats_df = pd.DataFrame(summary_stats)
        output_path = self.features_dir / output_file
        stats_df.to_csv(output_path, index=False)
        
        print(f"✅ 汇总统计已保存至: {output_path}")
        return stats_df

def main():
    """主函数 - 演示分析功能"""
    
    # 初始化分析器
    analyzer = WindowedEEGAnalyzer("../EEG_Features_Results_Windowed")
    
    # 尝试加载单个文件进行快速查看
    single_file = "G201601111533_zhujinxue_BeforeTask_GRA_F_Afterdis_0_windowed_features.csv"
    
    if analyzer.load_single_file_features(single_file):
        print("\\n" + "="*60)
        print("🔍 快速数据查看")
        print("="*60)
        
        # 显示数据概览
        analyzer.show_data_overview()
        
        # 绘制时间序列 (前2个通道, 前3个功率特征)
        print("\\n📈 绘制时间序列图...")
        analyzer.plot_time_series_features(
            channels=['O2', 'O1'], 
            features=['total_power', 'delta_power', 'alpha_power']
        )
        
        # 通道比较 (第一个窗口)
        print("\\n📊 绘制通道比较图...")
        analyzer.plot_channel_comparison(feature='total_power', window_index=0)
        
        # 特征相关性
        print("\\n🔗 绘制特征相关性图...")
        analyzer.plot_feature_correlation(window_index=0)
        
        # 导出统计
        print("\\n📝 导出汇总统计...")
        stats_df = analyzer.export_summary_stats()
        print(stats_df.head())
        
    else:
        print("❌ 无法加载数据文件")

if __name__ == "__main__":
    main()
