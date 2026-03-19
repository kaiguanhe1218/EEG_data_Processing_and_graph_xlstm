#!/usr/bin/env python3
"""
快速查看滑动窗口特征提取结果
"""

import pandas as pd
import numpy as np
from pathlib import Path

def quick_view_results():
    """快速查看处理结果"""
    
    results_dir = Path("../EEG_Features_Results_Windowed")
    
    print("="*70)
    print("🔍 EEG滑动窗口特征提取结果快速查看")
    print("="*70)
    
    # 检查结果目录
    if not results_dir.exists():
        print("❌ 结果目录不存在")
        return
    
    # 列出所有CSV文件
    csv_files = list(results_dir.glob("*.csv"))
    feature_files = [f for f in csv_files if 'windowed_features.csv' in f.name]
    summary_files = [f for f in csv_files if 'summary' in f.name]
    
    print(f"📁 结果目录: {results_dir}")
    print(f"📄 特征文件数量: {len(feature_files)}")
    print(f"📊 汇总文件数量: {len(summary_files)}")
    
    # 显示文件列表 (前10个)
    print(f"\\n📋 特征文件列表 (前10个):")
    for i, file in enumerate(feature_files[:10]):
        print(f"  {i+1:2d}. {file.name}")
    if len(feature_files) > 10:
        print(f"       ... 还有 {len(feature_files) - 10} 个文件")
    
    # 查看第一个特征文件的详细信息
    if feature_files:
        print(f"\\n" + "="*50)
        print(f"📊 样例文件分析: {feature_files[0].name}")
        print("="*50)
        
        df = pd.read_csv(feature_files[0])
        
        # 基本信息
        print(f"📈 数据形状: {df.shape}")
        print(f"🧠 通道数: {df['channel'].nunique()}")
        print(f"🪟 窗口数: {df['window_index'].nunique()}")
        print(f"⏰ 时间范围: {df['start_time'].min():.1f} - {df['end_time'].max():.1f} 秒")
        
        # 通道列表
        channels = df['channel'].unique()
        print(f"\\n📡 EEG通道 ({len(channels)}个):")
        print("  ", end="")
        for i, ch in enumerate(channels):
            print(f"{ch:>4}", end="")
            if (i + 1) % 15 == 0:  # 每15个换行
                print("\\n  ", end="")
        print()
        
        # 特征统计
        feature_cols = [col for col in df.columns 
                       if col not in ['filename', 'subject_id', 'condition', 'trial_id', 
                                    'channel', 'window_index', 'start_time', 'end_time']]
        
        print(f"\\n🎯 特征列表 ({len(feature_cols)}个):")
        power_features = [f for f in feature_cols if 'power' in f]
        ratio_features = [f for f in feature_cols if 'ratio' in f]
        other_features = [f for f in feature_cols if f not in power_features + ratio_features]
        
        print(f"  功率特征: {power_features}")
        print(f"  比例特征: {ratio_features}")
        print(f"  其他特征: {other_features[:5]}{'...' if len(other_features) > 5 else ''}")
        
        # 数据质量检查
        print(f"\\n✅ 数据质量检查:")
        # 检查是否有零值
        zero_counts = {}
        for col in power_features:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                zero_counts[col] = zero_count
        
        if zero_counts:
            print(f"⚠️  发现零值特征: {zero_counts}")
        else:
            print("✅ 所有功率特征均为非零值")
        
        # 显示一些统计信息
        print(f"\\n📊 主要特征统计:")
        main_features = ['total_power', 'delta_power', 'theta_power', 'alpha_power', 'beta_power']
        for feature in main_features:
            if feature in df.columns:
                values = df[feature]
                print(f"  {feature:12}: mean={values.mean():8.2f}, std={values.std():8.2f}, "
                      f"range=[{values.min():8.2f}, {values.max():8.2f}]")
        
        # 显示几行样例数据
        print(f"\\n📋 样例数据 (前3行):")
        display_cols = ['channel', 'window_index', 'start_time', 'end_time'] + main_features[:3]
        print(df[display_cols].head(3).to_string(index=False))
        
    # 检查汇总统计文件
    stats_file = results_dir / "windowed_features_summary_stats.csv"
    if stats_file.exists():
        print(f"\\n" + "="*50)
        print("📈 汇总统计信息")
        print("="*50)
        
        stats_df = pd.read_csv(stats_file)
        print(f"📊 统计特征数: {len(stats_df)}")
        
        # 显示功率特征的统计
        power_stats = stats_df[stats_df['feature'].str.contains('power')]
        print(f"\\n⚡ 功率特征统计:")
        for _, row in power_stats.iterrows():
            print(f"  {row['feature']:12}: mean={row['mean']:8.2f}, std={row['std']:8.2f}")
    
    print(f"\\n" + "="*70)
    print("✅ 特征提取结果查看完成")
    print("="*70)

if __name__ == "__main__":
    quick_view_results()
