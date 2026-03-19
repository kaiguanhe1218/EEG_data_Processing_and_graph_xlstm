"""
EEG特征提取配置文件
"""

# 数据处理配置
DATA_CONFIG = {
    # 输入数据路径
    'input_dir': '/mnt/d/project2025/BeforeTask(放电前后数据)',
    
    # 输出特征路径
    'output_dir': '/mnt/d/project2025/BeforeTask_features(放电前后数据)',
    
    # 采样率
    'sampling_rate': 500.0,
    
    # PSD计算参数
    'psd_params': {
        'nperseg': 512,  # 窗口长度
        'overlap': 0.5,  # 重叠比例
    }
}

# 频带定义
FREQUENCY_BANDS = {
    'standard_bands': [
        ("Total", 0.5, 100.),
        ("Delta", 0.5, 4.),
        ("Theta", 4.0, 8.),
        ("Alpha", 8.0, 12.),
        ("Beta", 12.0, 30.),
        ("Gamma", 30., 100.),
    ],
    
    'extended_bands': [
        ("Low Total", 0.5, 40.),
        ("Low Delta", 0.5, 2.),
        ("High Delta", 2., 4.),
        ("High Alpha", 10., 14.),
        ("Low Beta", 12., 20.),
        ("High Beta", 20., 30.),
        ("Low Gamma", 30., 47.),
    ],
    
    'clinical_bands': [
        ("CSI_beta", 11., 21.),
        ("CSI_alpha", 6, 12.),
        ("CSI_gamma", 30., 42.5),
        ("BIS_beta", 11., 20.),
        ("BIS_gamma", 30., 47.),
    ]
}

# 特征提取配置
FEATURE_CONFIG = {
    # 是否计算扩展特征
    'compute_extended_features': True,
    
    # 是否计算临床指标
    'compute_clinical_indices': True,
    
    # 熵计算配置
    'entropy_params': {
        'state_entropy_range': (0.8, 32),
        'response_entropy_range': (0.8, 47),
        'normalize': True
    },
    
    # SEF计算百分位
    'sef_percentiles': [0.05, 0.5, 0.9, 0.95],
    
    # 平均频率计算范围
    'average_frequency_ranges': {
        'alpha': (8.0, 12.0),
        'beta': (12.0, 30.0),
        'theta': (4.0, 8.0)
    }
}

# 输出配置
OUTPUT_CONFIG = {
    # 保存格式
    'save_formats': ['csv', 'pickle'],
    
    # 是否生成单个文件的特征
    'save_individual_files': True,
    
    # 是否生成汇总报告
    'generate_summary': True,
    
    # 是否生成统计报告
    'generate_statistics': True,
    
    # 数值精度
    'decimal_places': 4
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file_logging': True,
    'log_filename': 'eeg_feature_extraction.log'
}

# 质量控制配置
QUALITY_CONFIG = {
    # 信号质量检查
    'check_signal_quality': True,
    
    # 最小信号变化阈值
    'min_signal_variance': 1e-12,
    
    # 最大信号幅值（微伏）
    'max_signal_amplitude': 1000,
    
    # 是否移除基线漂移
    'remove_baseline': True,
    
    # 是否进行滤波
    'apply_filtering': False,
    'filter_params': {
        'low_cutoff': 0.5,
        'high_cutoff': 100,
        'filter_order': 4
    }
}
