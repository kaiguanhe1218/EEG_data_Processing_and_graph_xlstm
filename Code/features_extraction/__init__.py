"""
EEG特征提取包初始化文件
"""

# 导入核心模块
try:
    from eeg_feature_extraction import EEGFeatureExtractor, EEGFeatures
    __all__ = ['EEGFeatureExtractor', 'EEGFeatures']
except ImportError:
    __all__ = []

# 尝试导入分析模块
try:
    from eeg_feature_analysis import EEGFeatureAnalyzer
    __all__.append('EEGFeatureAnalyzer')
except ImportError:
    # 如果缺少可视化依赖，跳过分析模块
    pass

__version__ = "1.0.0"
__author__ = "EEG Feature Extraction System"
