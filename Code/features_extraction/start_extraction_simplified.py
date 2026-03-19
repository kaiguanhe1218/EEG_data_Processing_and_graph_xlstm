#!/usr/bin/env python3
"""
启动完整的EEG特征提取 - 简化版本
所有数据都是V单位，直接转换为μV
"""

import os
import sys
import pandas as pd
from pathlib import Path
import logging

# 添加当前目录到Python路径
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from eeg_feature_extraction import EEGFeatureExtractor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数：处理所有TSV文件"""
    print("🚀 EEG特征提取 - 简化版本 (V→μV直接转换)")
    print("="*60)
    
    # 输入和输出路径
    input_dir = "/mnt/d/project2025/BeforeTask(放电前后数据)"
    output_dir = "/mnt/d/project2025/EEG_Features_Results"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 找到所有TSV文件
    tsv_files = list(Path(input_dir).glob("*.tsv"))
    
    if not tsv_files:
        print("❌ 未找到TSV文件")
        return
    
    print(f"📁 找到 {len(tsv_files)} 个TSV文件")
    
    # 创建特征提取器
    extractor = EEGFeatureExtractor()
    
    # 存储所有结果
    all_results = []
    processed_count = 0
    error_count = 0
    
    # 处理每个文件
    for i, file_path in enumerate(tsv_files, 1):
        try:
            print(f"\n📄 [{i}/{len(tsv_files)}] 处理: {file_path.name}")
            
            # 加载数据
            eeg_data, channel_names, meta_info = extractor.load_tsv_file(str(file_path))
            
            # 解析文件名
            subject_id, condition, base_name, discharge_point = extractor.parse_filename(file_path.name)
            
            # 处理每个通道
            file_results = []
            for j, channel_name in enumerate(channel_names):
                signal_data = eeg_data[:, j]
                features = extractor.extract_features_from_channel(signal_data)
                
                # 创建结果记录
                result = features.to_dict()
                result.update({
                    'file': file_path.name,
                    'subject_id': subject_id,
                    'condition': condition,
                    'discharge_point': discharge_point,
                    'channel': channel_name,
                    'conversion_factor': extractor.conversion_factor
                })
                
                file_results.append(result)
            
            all_results.extend(file_results)
            processed_count += 1
            
            # 显示处理状态
            print(f"✅ 成功处理 {len(channel_names)} 个通道")
            
        except Exception as e:
            error_count += 1
            logger.error(f"处理文件 {file_path.name} 失败: {str(e)}")
            print(f"❌ 跳过文件: {file_path.name}")
    
    # 保存结果
    if all_results:
        print(f"\n💾 保存结果...")
        results_df = pd.DataFrame(all_results)
        
        # 保存总结果
        summary_file = os.path.join(output_dir, "eeg_features_summary.csv")
        results_df.to_csv(summary_file, index=False)
        print(f"✅ 总结果已保存: {summary_file}")
        
        # 保存按条件分组的结果
        for condition in results_df['condition'].unique():
            condition_df = results_df[results_df['condition'] == condition]
            condition_file = os.path.join(output_dir, f"eeg_features_{condition}.csv")
            condition_df.to_csv(condition_file, index=False)
            print(f"✅ {condition} 结果已保存: {condition_file}")
        
        # 统计信息
        total_features = len(all_results)
        total_channels = len(results_df['channel'].unique())
        
        print(f"\n📊 处理统计:")
        print(f"  成功处理文件: {processed_count}")
        print(f"  失败文件: {error_count}")
        print(f"  总特征记录: {total_features}")
        print(f"  总通道数: {total_channels}")
        print(f"  转换因子: {extractor.conversion_factor} (V→μV)")
        
        # 检查数据质量
        power_cols = [col for col in results_df.columns if col.endswith('_power')]
        zero_counts = (results_df[power_cols] == 0).sum()
        
        if zero_counts.sum() == 0:
            print(f"\n✅ 数据质量良好：所有功率特征都有非零值！")
        else:
            print(f"\n⚠️ 数据质量检查：发现部分零值")
            for col, count in zero_counts.items():
                if count > 0:
                    print(f"  {col}: {count} 个零值")
    
    else:
        print("❌ 没有成功处理的结果")
    
    print(f"\n🎉 特征提取完成！")

if __name__ == "__main__":
    main()
