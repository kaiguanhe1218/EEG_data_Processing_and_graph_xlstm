#!/usr/bin/env python3
"""
EEG特征提取 - 滑动窗口版本
"""

import os
import glob
import pandas as pd
from pathlib import Path
from eeg_feature_extraction import EEGFeatureExtractor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数 - 使用滑动窗口提取特征"""
    
    # 配置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data/BeforeTask(未放电)"
    output_dir = base_dir / "EEG_Features_Results_Windowed_nodis"
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有TSV文件
    tsv_files = list(data_dir.glob("*.tsv"))
    logger.info(f"找到 {len(tsv_files)} 个TSV文件")
    
    # 初始化特征提取器
    # 参数说明:
    # - target_length: 20000 (重采样到20000点，10秒，2000Hz)
    # - window_size: 2000 (0.1秒窗口)
    # - step_size: 1800 (10%重叠)
    extractor = EEGFeatureExtractor(
        target_length=2000,
        window_size=200,
        step_size=100,
        psd_nperseg = 200
    )
    
    all_features = []
    
    # 处理每个文件
    for i, file_path in enumerate(tsv_files):
        logger.info(f"\\n处理文件 {i+1}/{len(tsv_files)}: {file_path.name}")
        
        try:
            # 提取特征
            features_df = extractor.extract_file_features(str(file_path))
            all_features.append(features_df)
            
            # 保存单个文件的特征
            output_file = output_dir / f"{file_path.stem}_windowed_features.csv"
            features_df.to_csv(output_file, index=False)
            logger.info(f"保存特征文件: {output_file.name}")
            
        except Exception as e:
            logger.error(f"处理文件 {file_path.name} 失败: {str(e)}")
            continue
    
    # 合并所有特征
    if all_features:
        logger.info("\\n合并所有特征...")
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # 保存合并后的特征
        summary_file = output_dir / "all_windowed_features_summary.csv"
        combined_features.to_csv(summary_file, index=False)
        
        logger.info(f"\\n✅ 特征提取完成!")
        logger.info(f"📊 总特征数量: {len(combined_features)}")
        logger.info(f"📊 文件数量: {len(all_features)}")
        logger.info(f"📊 每文件平均窗口数: {len(combined_features) // len(all_features)}")
        logger.info(f"📁 结果保存在: {output_dir}")
        logger.info(f"📄 汇总文件: {summary_file.name}")
        
        # 显示特征概览
        print("\\n📈 特征概览:")
        print(f"形状: {combined_features.shape}")
        print(f"通道数: {combined_features['channel'].nunique()}")
        print(f"受试者数: {combined_features['subject_id'].nunique()}")
        print(f"窗口数范围: {combined_features['window_index'].min()} - {combined_features['window_index'].max()}")
        
        # 检查特征值
        feature_cols = [col for col in combined_features.columns 
                       if col.endswith(('_power', '_ratio', '_entropy', 'SEF_'))]
        print(f"\\n🎯 主要特征统计 (前5个):")
        for col in feature_cols[:5]:
            values = pd.to_numeric(combined_features[col], errors='coerce')
            print(f"  {col}: mean={values.mean():.6f}, std={values.std():.6f}")
    
    else:
        logger.error("❌ 没有成功处理任何文件!")

if __name__ == "__main__":
    main()
