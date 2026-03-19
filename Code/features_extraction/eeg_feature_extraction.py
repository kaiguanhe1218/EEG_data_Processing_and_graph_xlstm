#!/usr/bin/env python3
"""
修复后的EEG特征提取器
主要修复：
1. 添加数据单位转换
2. 修改数值精度处理
3. 使用科学计数法保存小数值
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import simpson
from scipy.stats import entropy
import logging
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EEGFeatures:
    """EEG特征数据类"""
    # 基础功率特征
    total_power: float = 0.0
    delta_power: float = 0.0
    theta_power: float = 0.0
    alpha_power: float = 0.0
    beta_power: float = 0.0
    gamma_power: float = 0.0
    
    # 相对功率特征
    delta_ratio: float = 0.0
    theta_ratio: float = 0.0
    alpha_ratio: float = 0.0
    beta_ratio: float = 0.0
    gamma_ratio: float = 0.0
    
    # 功率比值特征
    ADR: float = 0.0  # Alpha/Delta ratio
    ATR: float = 0.0  # Alpha/Theta ratio
    ABR: float = 0.0  # Alpha/Beta ratio
    GAR: float = 0.0  # Gamma/Alpha ratio
    GBR: float = 0.0  # Gamma/Beta ratio
    GDR: float = 0.0  # Gamma/Delta ratio
    GTR: float = 0.0  # Gamma/Theta ratio
    
    # 频谱边缘频率
    SEF_5: float = 0.0   # 5%功率频率
    SEF_50: float = 0.0  # 50%功率频率
    SEF_90: float = 0.0  # 90%功率频率
    SEF_95: float = 0.0  # 95%功率频率
    
    # 熵特征
    spectral_entropy: float = 0.0
    state_entropy: float = 0.0
    response_entropy: float = 0.0
    
    # 其他特征
    low_total_power: float = 0.0
    delta_proportion: float = 0.0
    theta_proportion: float = 0.0
    alpha_proportion: float = 0.0
    beta_proportion: float = 0.0
    gamma_proportion: float = 0.0
    alpha_average_frequency: float = 0.0
    weighted_alpha_average_frequency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，使用科学计数法保存小数值"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)):
                # 对于极小的数值，使用科学计数法
                if abs(value) < 1e-6 and value != 0:
                    result[key] = f"{value:.6e}"
                else:
                    result[key] = round(value, 6)  # 增加精度到6位
            else:
                result[key] = value
        return result


class EEGFeatureExtractor:
    """修复后的EEG特征提取器 - 支持滑动窗口处理"""
    
    def __init__(self, target_length: int = 2000, window_size: int = 200, step_size: int = 100, psd_nperseg: int | None = None):
        self.target_length = target_length  # 重采样目标长度
        self.target_fs = target_length / 10.0  # 目标采样率 200Hz
        self.window_size = window_size  # 窗口大小 (1秒)
        self.step_size = step_size      # 步长 (0.5秒)
        self.conversion_factor = 1e6    # 固定转换因子：V转μV
        self.psd_nperseg = psd_nperseg  # Welch变换段长度，可显式控制频率分辨率
        
        # 定义频带 (适合200Hz采样率，Nyquist频率100Hz)
        self.freq_bands = [
            ("Delta", 0.5, 4.0),
            ("Theta", 4.0, 8.0),
            ("Alpha", 8.0, 12.0),
            ("Beta", 12.0, 30.0),
            ("Gamma", 30.0, 50.0),  # 降低到50Hz以适应100Hz Nyquist频率
            ("Total", 0.5, 50.0)    # 总频段也调整到50Hz
        ]
    
    def parse_filename(self, filename: str) -> Tuple[str, str, str, int]:
        """解析文件名"""
        # 移除文件扩展名
        base_name = os.path.splitext(filename)[0]
        
        # 解析格式: 编号_姓名_BeforeTask_GRA_F_数据标识_放电点
        parts = base_name.split('_')
        if len(parts) >= 6:
            subject_id = f"{parts[0]}_{parts[1]}"
            condition = parts[5]  # Beforedis 或 Afterdis
            discharge_point = int(parts[6]) if len(parts) > 6 else 0
            return subject_id, condition, base_name, discharge_point
        else:
            raise ValueError(f"无法解析文件名: {filename}")
    
    def convert_units(self, eeg_data: np.ndarray) -> np.ndarray:
        """直接转换数据单位：V转μV"""
        converted_data = eeg_data * self.conversion_factor
        logger.info(f"已将数据从V转换为μV (×{self.conversion_factor})")
        return converted_data
    
    def resample_data(self, data: np.ndarray, original_length: int, target_length: int) -> np.ndarray:
        """重采样数据到目标长度"""
        from scipy.interpolate import interp1d
        
        # 原始时间轴
        original_time = np.linspace(0, 10, original_length)
        # 目标时间轴
        target_time = np.linspace(0, 10, target_length)
        
        # 对每个通道进行插值重采样
        resampled_data = np.zeros((target_length, data.shape[1]))
        
        for ch in range(data.shape[1]):
            # 使用线性插值
            f = interp1d(original_time, data[:, ch], kind='linear')
            resampled_data[:, ch] = f(target_time)
        
        return resampled_data
    
    def load_tsv_file(self, file_path: str) -> Tuple[np.ndarray, List[str], Dict]:
        """加载TSV文件并进行预处理"""
        try:
            data = pd.read_csv(file_path, sep='\t')
            
            # 获取时间信息
            times = data['times'].values
            original_duration = times[-1] - times[0]
            original_length = len(data)
            original_fs = original_length / original_duration
            
            logger.info(f"原始数据: 长度={original_length}, 时长={original_duration:.1f}s, 采样率={original_fs:.1f}Hz")
            
            # 排除非EEG通道
            exclude_cols = ['epochs_id', 'times', 'id']
            eeg_columns = [col for col in data.columns if col not in exclude_cols]
            
            # 提取EEG数据
            eeg_data = data[eeg_columns].values.astype(float)
            
            # 重采样到目标长度
            eeg_data = self.resample_data(eeg_data, original_length, self.target_length)
            logger.info(f"重采样后: 长度={self.target_length}, 采样率={self.target_fs:.1f}Hz")
            
            # 转换单位 V -> μV
            eeg_data = self.convert_units(eeg_data)
            
            # 提取元信息
            file_name = os.path.basename(file_path)
            subject_id, condition, full_name, trial_id = self.parse_filename(file_name)
            meta_info = {
                'filename': file_name,
                'subject_id': subject_id,
                'condition': condition,
                'trial_id': trial_id,
                'epochs_id': data['epochs_id'].iloc[0] if 'epochs_id' in data.columns else None,
                'original_length': original_length,
                'original_fs': original_fs,
                'target_length': self.target_length,
                'target_fs': self.target_fs,
                'conversion_factor': self.conversion_factor
            }
            
            logger.info(f"加载数据: {eeg_data.shape}, 通道数: {len(eeg_columns)}")
            logger.info(f"数据范围: [{eeg_data.min():.2f}, {eeg_data.max():.2f}] μV")
            
            return eeg_data, eeg_columns, meta_info
            
        except Exception as e:
            logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            raise
    
    def compute_psd(self, signal_data: np.ndarray, nperseg: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算功率谱密度"""
        if signal_data.ndim > 1:
            signal_data = signal_data.flatten()
        
        # 对于滑动窗口，使用较大的nperseg以获得更好的频率分辨率
        # 但不能超过信号长度
        if nperseg is None:
            if self.psd_nperseg is not None:
                nperseg = min(self.psd_nperseg, len(signal_data))
            else:
                # 使用信号长度的一半，但最少64个点
                nperseg = min(max(len(signal_data)//2, 64), len(signal_data))
        
        # 确保nperseg不超过信号长度
        nperseg = min(nperseg, len(signal_data))
        
        freqs, psd = signal.welch(
            signal_data, 
            fs=self.target_fs,  # 使用目标采样率 200Hz
            nperseg=nperseg
        )
        
        if psd.ndim > 1:
            psd = psd.flatten()
            
        return freqs, psd
    
    def get_band_power(self, psd: np.ndarray, freqs: np.ndarray, freq_bands: List) -> Dict:
        """计算频带功率"""
        band_powers = {}
        nyquist_freq = freqs[-1] if len(freqs) > 0 else 0.0
        min_freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else float('inf')
        
        if psd.ndim > 1:
            psd = psd.flatten()
        
        # 打印频率范围信息（仅第一次）
        if not hasattr(self, '_freq_info_printed'):
            logger.info(f"频率范围: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz, 分辨率: {freqs[1]-freqs[0]:.2f} Hz")
            self._freq_info_printed = True
        
        for band_name, fmin, fmax in freq_bands:
            # 确保频率范围在有效范围内
            fmin = max(fmin, freqs[0])
            fmax = min(fmax, freqs[-1])

            # 当频带宽度过小而低于频率分辨率时，直接跳过该频带
            if fmax - fmin <= min_freq_resolution:
                logger.warning(
                    "频带 %s (%.1f-%.1fHz) 与当前频率分辨率 %.2fHz 不匹配，已跳过",
                    band_name, fmin, fmax, min_freq_resolution
                )
                continue
            
            # 找到频率范围内的索引
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            
            if np.any(idx):
                band_psd = psd[idx]
                band_freqs = freqs[idx]
                
                # 使用梯形积分
                if len(band_freqs) > 1:
                    band_power = np.trapz(band_psd, band_freqs)
                else:
                    band_power = band_psd[0] * (freqs[1] - freqs[0])
                
                band_powers[band_name] = max(band_power, 1e-12)  # 防止完全为0
            else:
                logger.warning(
                    "频带 %s (%.1f-%.1fHz) 没有找到对应的频率点，Nyquist=%.1fHz",
                    band_name, fmin, fmax, nyquist_freq
                )
                continue
        
        return band_powers
    
    def compute_spectral_edge_frequency(self, psd: np.ndarray, freqs: np.ndarray, percentage: float) -> float:
        """计算频谱边缘频率"""
        if psd.ndim > 1:
            psd = psd.flatten()
        
        total_power = np.sum(psd)
        
        if total_power == 0:
            return 0.0
        
        cumulative_power = np.cumsum(psd)
        normalized_cum_power = cumulative_power / total_power
        
        target_power = percentage
        idx = np.where(normalized_cum_power >= target_power)[0]
        
        if len(idx) > 0:
            return freqs[idx[0]]
        else:
            return freqs[-1]
    
    def compute_spectral_entropy(self, psd: np.ndarray) -> float:
        """计算频谱熵"""
        if psd.ndim > 1:
            psd = psd.flatten()
        
        if np.sum(psd) == 0:
            return 0.0
        
        normalized_psd = psd / np.sum(psd)
        normalized_psd = normalized_psd[normalized_psd > 0]
        
        if len(normalized_psd) == 0:
            return 0.0
        
        return entropy(normalized_psd)

    def compute_state_entropy(self, signal: np.ndarray) -> float:
        """计算状态熵
        状态熵用于衡量信号的复杂度，使用信号的振幅分布"""
        if signal.ndim > 1:
            signal = signal.flatten()
            
        # 使用Freedman-Diaconis规则确定bin数
        n_samples = len(signal)
        iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
        bin_width = 2 * iqr / (n_samples ** (1/3))  # Freedman-Diaconis规则
        
        if bin_width == 0:  # 处理常量信号
            return 0.0
        
        n_bins = int(np.ceil((np.max(signal) - np.min(signal)) / bin_width))
        n_bins = max(min(n_bins, 100), 10)  # 限制bin数量在合理范围内
        
        # 计算直方图
        hist, _ = np.histogram(signal, bins=n_bins, density=True)
        hist = hist[hist > 0]  # 移除零频率
        
        # 计算熵
        if len(hist) == 0:
            return 0.0
            
        return entropy(hist)
        
    def compute_response_entropy(self, signal: np.ndarray) -> float:
        """计算响应熵
        响应熵考虑信号的动态特性，使用信号的差分分布"""
        if signal.ndim > 1:
            signal = signal.flatten()
            
        # 计算信号的一阶差分
        diff_signal = np.diff(signal)
        
        # 使用与状态熵相同的方法计算bin数
        n_samples = len(diff_signal)
        iqr = np.percentile(diff_signal, 75) - np.percentile(diff_signal, 25)
        bin_width = 2 * iqr / (n_samples ** (1/3))
        
        if bin_width == 0:  # 处理常量信号
            return 0.0
        
        n_bins = int(np.ceil((np.max(diff_signal) - np.min(diff_signal)) / bin_width))
        n_bins = max(min(n_bins, 100), 10)  # 限制bin数量
        
        # 计算差分信号的直方图
        hist, _ = np.histogram(diff_signal, bins=n_bins, density=True)
        hist = hist[hist > 0]  # 移除零频率
        
        # 计算熵
        if len(hist) == 0:
            return 0.0
            
        return entropy(hist)
    
    def extract_features_from_channel(self, signal_data: np.ndarray) -> EEGFeatures:
        """从单个通道提取特征"""
        features = EEGFeatures()
        
        if signal_data.ndim > 1:
            signal_data = signal_data.flatten()
        
        freqs, psd = self.compute_psd(signal_data)
        band_powers = self.get_band_power(psd, freqs, self.freq_bands)
        
        # 基础功率特征 (不再四舍五入，保持原始精度)
        features.total_power = band_powers.get("Total", 0.0)
        features.delta_power = band_powers.get("Delta", 0.0)
        features.theta_power = band_powers.get("Theta", 0.0)
        features.alpha_power = band_powers.get("Alpha", 0.0)
        features.beta_power = band_powers.get("Beta", 0.0)
        features.gamma_power = band_powers.get("Gamma", 0.0)
        
        # 计算相对功率特征
        if features.total_power > 0:
            features.delta_ratio = 100 * features.delta_power / features.total_power
            features.theta_ratio = 100 * features.theta_power / features.total_power
            features.alpha_ratio = 100 * features.alpha_power / features.total_power
            features.beta_ratio = 100 * features.beta_power / features.total_power
            features.gamma_ratio = 100 * features.gamma_power / features.total_power
        
        # 计算功率比值特征
        if features.delta_power > 0:
            features.ADR = features.alpha_power / features.delta_power
            features.GDR = features.gamma_power / features.delta_power
        
        if features.theta_power > 0:
            features.ATR = features.alpha_power / features.theta_power
            features.GTR = features.gamma_power / features.theta_power
        
        if features.beta_power > 0:
            features.ABR = features.alpha_power / features.beta_power
            features.GBR = features.gamma_power / features.beta_power
        
        if features.alpha_power > 0:
            features.GAR = features.gamma_power / features.alpha_power
        
        # 频谱边缘频率
        features.SEF_5 = self.compute_spectral_edge_frequency(psd, freqs, 0.05)
        features.SEF_50 = self.compute_spectral_edge_frequency(psd, freqs, 0.5)
        features.SEF_90 = self.compute_spectral_edge_frequency(psd, freqs, 0.9)
        features.SEF_95 = self.compute_spectral_edge_frequency(psd, freqs, 0.95)
        
        # 熵特征
        features.spectral_entropy = self.compute_spectral_entropy(psd)
        features.state_entropy = self.compute_state_entropy(signal_data)
        features.response_entropy = self.compute_response_entropy(signal_data)
        
        # 计算proportion特征 (与ratio类似，但用小数表示)
        if features.total_power > 0:
            features.delta_proportion = features.delta_power / features.total_power
            features.theta_proportion = features.theta_power / features.total_power
            features.alpha_proportion = features.alpha_power / features.total_power
            features.beta_proportion = features.beta_power / features.total_power
            features.gamma_proportion = features.gamma_power / features.total_power
        
        # 计算low_total_power特征 (低频段功率，通常是0.5-30Hz)
        low_freq_bands = [("LowFreq", 0.5, 30.0)]
        low_band_powers = self.get_band_power(psd, freqs, low_freq_bands)
        features.low_total_power = low_band_powers["LowFreq"]
        
        # 计算alpha平均频率
        alpha_idx = np.logical_and(freqs >= 8.0, freqs <= 12.0)
        if np.any(alpha_idx):
            alpha_freqs = freqs[alpha_idx]
            alpha_psd = psd[alpha_idx]
            if np.sum(alpha_psd) > 0:
                # 简单平均频率
                features.alpha_average_frequency = np.mean(alpha_freqs)
                # 加权平均频率
                features.weighted_alpha_average_frequency = np.average(alpha_freqs, weights=alpha_psd)
        
        return features

    def sliding_window_features(self, eeg_data: np.ndarray, channel_names: List[str], meta_info: Dict) -> pd.DataFrame:
        """使用滑动窗口提取特征"""
        all_features = []
        
        # 计算窗口数量
        n_windows = (self.target_length - self.window_size) // self.step_size + 1
        logger.info(f"滑动窗口处理: 窗口大小={self.window_size}, 步长={self.step_size}, 窗口数量={n_windows}")
        
        for ch_idx, channel_name in enumerate(channel_names):
            logger.info(f"处理通道 {ch_idx+1}/{len(channel_names)}: {channel_name}")
            
            channel_data = eeg_data[:, ch_idx]
            
            for win_idx in range(n_windows):
                # 计算窗口范围
                start_idx = win_idx * self.step_size
                end_idx = start_idx + self.window_size
                
                # 提取窗口数据
                window_data = channel_data[start_idx:end_idx]
                
                # 计算时间信息
                start_time = start_idx / self.target_fs
                end_time = end_idx / self.target_fs
                
                # 提取特征
                features = self.extract_features_from_channel(window_data)
                feature_dict = features.to_dict()
                
                # 添加窗口和通道信息
                feature_dict.update({
                    'window_index': win_idx,
                    'start_time': round(start_time, 3),
                    'end_time': round(end_time, 3),
                    'channel': channel_name,
                    'filename': meta_info['filename'],
                    'subject_id': meta_info['subject_id'],
                    'condition': meta_info['condition'],
                    'trial_id': meta_info['trial_id']
                })
                
                all_features.append(feature_dict)
        
        # 转换为DataFrame
        features_df = pd.DataFrame(all_features)
        
        # 重新排序列，将标识信息放在前面
        id_cols = ['filename', 'subject_id', 'condition', 'trial_id', 'channel', 'window_index', 'start_time', 'end_time']
        feature_cols = [col for col in features_df.columns if col not in id_cols]
        features_df = features_df[id_cols + feature_cols]
        
        logger.info(f"提取完成: {len(features_df)} 个特征窗口")
        return features_df
    
    def extract_file_features(self, file_path: str) -> pd.DataFrame:
        """从单个文件提取所有滑动窗口特征"""
        try:
            # 加载和预处理数据
            eeg_data, channel_names, meta_info = self.load_tsv_file(file_path)
            
            # 使用滑动窗口提取特征
            features_df = self.sliding_window_features(eeg_data, channel_names, meta_info)
            
            return features_df
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 失败: {str(e)}")
            raise


def main():
    """主函数：测试特征提取器"""
    print("🧪 EEG特征提取器测试")
    print("="*50)
    
    # 测试文件
    test_file = "/mnt/d/project2025/BeforeTask(放电前后数据)/G201601111533_zhujinxue_BeforeTask_GRA_F_Afterdis_0.tsv"
    
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    try:
        # 创建特征提取器
        extractor = EEGFeatureExtractor()
        
        # 加载数据
        eeg_data, channel_names, meta_info = extractor.load_tsv_file(test_file)
        
        # 提取第一个通道的特征
        channel_name = channel_names[0]
        signal_data = eeg_data[:, 0]
        
        print(f"\n📊 处理通道: {channel_name}")
        print(f"📊 转换因子: {extractor.conversion_factor}")
        
        # 提取特征
        features = extractor.extract_features_from_channel(signal_data)
        
        # 显示特征
        print(f"\n🎯 提取的特征:")
        feature_dict = features.to_dict()
        
        for key, value in feature_dict.items():
            if key.endswith('_power'):
                print(f"  {key}: {value}")
        
        print(f"\n🔍 其他关键特征:")
        print(f"  delta_ratio: {feature_dict['delta_ratio']}")
        print(f"  alpha_ratio: {feature_dict['alpha_ratio']}")
        print(f"  spectral_entropy: {feature_dict['spectral_entropy']}")
        print(f"  SEF_90: {feature_dict['SEF_90']}")
        
        # 检查是否还有0值
        power_features = ['total_power', 'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
        zero_features = [f for f in power_features if feature_dict[f] == 0 or feature_dict[f] == '0.000000e+00']
        
        if zero_features:
            print(f"\n❌ 仍有0值特征: {zero_features}")
        else:
            print(f"\n✅ 所有功率特征都有非零值!")
            
    except Exception as e:
        print(f"❌ 特征提取失败: {str(e)}")
        return
    
    print(f"\n✅ 修复完成！")

if __name__ == "__main__":
    main()
