# EEG特征提取系统使用说明

## 项目概述
start run "start_windowed_extraction"

本项目基于原始的`feature_extraction.py`代码，专门为处理`BeforeTask`文件夹中的TSV格式EEG数据而设计，用于提取各种频谱特征和临床指标。

## 文件结构

```
project2025/
├── BeforeTask(放电前后数据)/          # 原始TSV数据文件
├── BeforeTask_features(放电前后数据)/   # 特征提取代码目录
│   ├── eeg_feature_extraction.py      # 主要特征提取脚本
│   ├── eeg_feature_analysis.py        # 数据分析和可视化脚本
│   ├── config.py                      # 配置文件
│   ├── README.md                      # 使用说明（本文件）
│   └── requirements.txt               # 依赖包列表
└── EEG_Features_Results/               # 特征提取结果目录 (自动创建)
    ├── *_features.csv                  # 各文件的特征结果
    ├── all_features_summary.csv       # 汇总结果
    └── statistics_report.txt          # 统计报告
```

## 数据格式

### 输入数据格式
- **文件格式**: TSV (Tab-Separated Values)
- **文件命名规则**: `编号_姓名_BeforeTask_GRA_F_数据标识_放电点.tsv`
  - 例如: `G201601111533_zhujinxue_BeforeTask_GRA_F_Afterdis_0.tsv`
  - `数据标识`: `Beforedis`（放电前）或 `Afterdis`（放电后）
  - `放电点`: 数字，表示第几个放电节点

### 数据结构
- **列名**: `epochs_id`, `times`, `id`, 以及各EEG通道名
- **epochs_id**: 数据在原始数据中的位置
- **times**: 时间轴，以0开始，单位为秒
- **id**: 从0开始依次递增，与times对应
- **通道数据**: 各EEG通道的电压值（微伏）

## 安装依赖

```bash
pip install numpy pandas scipy matplotlib seaborn
```

或使用requirements.txt：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 特征提取

#### 基本使用
```python
from eeg_feature_extraction import EEGFeatureExtractor

# 创建特征提取器
extractor = EEGFeatureExtractor(sampling_rate=500.0)

# 处理整个目录
input_dir = "/path/to/BeforeTask(放电前后数据)"
output_dir = "/path/to/BeforeTask_features(放电前后数据)"
extractor.process_directory(input_dir, output_dir)
```

#### 命令行使用
```bash
cd BeforeTask_features(放电前后数据)
python3 start_extraction.py
```

**注意**: 特征提取结果将保存在 `../EEG_Features_Results/` 目录中，与代码文件分开存储。

### 2. 数据分析

```python
from eeg_feature_analysis import EEGFeatureAnalyzer

# 创建分析器，指向结果目录
analyzer = EEGFeatureAnalyzer("/mnt/d/project2025/EEG_Features_Results")

# 加载数据并生成报告
if analyzer.load_features():
    analyzer.generate_analysis_report("/mnt/d/project2025/EEG_Features_Results")
```

## 提取的特征

### 1. 基础功率特征
- **total_power**: 总功率 (0.5-100 Hz)
- **delta_power**: Delta波功率 (0.5-4 Hz)
- **theta_power**: Theta波功率 (4-8 Hz)
- **alpha_power**: Alpha波功率 (8-12 Hz)
- **beta_power**: Beta波功率 (12-30 Hz)
- **gamma_power**: Gamma波功率 (30-100 Hz)

### 2. 相对功率比
- **delta_ratio**: Delta波相对功率 (%)
- **theta_ratio**: Theta波相对功率 (%)
- **alpha_ratio**: Alpha波相对功率 (%)
- **beta_ratio**: Beta波相对功率 (%)
- **gamma_ratio**: Gamma波相对功率 (%)

### 3. 功率比率指标
- **ADR**: Alpha/Delta比率
- **ATR**: Alpha/Theta比率
- **ABR**: Alpha/Beta比率
- **GAR**: Gamma/Alpha比率
- **GBR**: Gamma/Beta比率
- **GDR**: Gamma/Delta比率
- **GTR**: Gamma/Theta比率

### 4. 频谱边缘频率 (SEF)
- **SEF_5**: 5%频谱边缘频率
- **SEF_50**: 50%频谱边缘频率 (中位频率)
- **SEF_90**: 90%频谱边缘频率
- **SEF_95**: 95%频谱边缘频率

### 5. 熵特征
- **spectral_entropy**: 频谱熵
- **state_entropy**: 状态熵 (0.8-32 Hz)
- **response_entropy**: 响应熵 (0.8-47 Hz)

### 6. 扩展特征
- **alpha_average_frequency**: Alpha频段平均频率
- **weighted_alpha_average_frequency**: 功率加权Alpha平均频率
- **delta_proportion**: Delta波在低频总功率中的比例
- **theta_proportion**: Theta波在低频总功率中的比例
- 等等...

## 输出文件

### 1. 单个文件特征
- **位置**: `/mnt/d/project2025/EEG_Features_Results/`
- **格式**: CSV格式
- **命名**: `原文件名_features.csv`
- **内容**: 该文件所有通道的特征

### 2. 汇总文件
- **all_features_summary.csv**: 所有文件的特征汇总
- **statistics_report.txt**: 统计报告

### 3. 分析报告
- **功率分布图**: `power_distribution.png`
- **条件比较图**: `condition_comparison.png`
- **相关性矩阵**: `correlation_matrix.png`
- **被试趋势图**: `subject_trends_*.png`

## 配置选项

在`config.py`中可以配置：
- 采样率设置
- 频带定义
- PSD计算参数
- 输出格式选项
- 日志级别
- 质量控制参数

## 注意事项

1. **内存使用**: 大文件可能消耗较多内存，建议分批处理
2. **文件格式**: 确保TSV文件格式正确，包含必要的列
3. **通道名称**: 系统会自动识别EEG通道，跳过非数值列
4. **数据质量**: 会自动检查和跳过全零或无效的通道
5. **错误处理**: 处理失败的文件会记录在`failed_files.txt`中

## 故障排除

### 常见问题

1. **文件加载失败**
   - 检查文件路径是否正确
   - 确认文件格式为TSV
   - 检查文件是否损坏

2. **特征提取失败**
   - 检查数据是否包含有效的数值
   - 确认采样率设置正确
   - 查看日志文件了解详细错误信息

3. **内存不足**
   - 减少nperseg参数
   - 分批处理文件
   - 增加系统内存

### 调试模式

在代码中设置更详细的日志级别：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

### 自定义频带
在`config.py`中添加新的频带定义：
```python
CUSTOM_BANDS = [
    ("Custom_Band", 10., 15.),
    # 添加更多自定义频带
]
```

### 新特征
在`EEGFeatureExtractor`类中添加新的特征计算方法，并更新`EEGFeatures`数据类。


