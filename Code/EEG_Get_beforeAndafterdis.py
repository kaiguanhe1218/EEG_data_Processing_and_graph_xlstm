import os
import mne
import numpy as np
import pandas as pd

def load_eeg_data(file_path):
    """读取EEG数据"""
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    
    # 如果数据长度超过500s，只取前500s
    if raw.times[-1] > 500:
        raw.crop(tmin=0, tmax=500)
    
    return raw

def get_events_and_epochs(raw):
    """获取事件和epochs数据"""
    events, event_id = mne.events_from_annotations(raw)
    
    # 只保留需要的事件标签
    needed_events = {k: v for k, v in event_id.items() if k in ['100005', 'dis']}
    
    # 创建epochs
    epochs_all = mne.Epochs(raw, events=events[events[:, 2] == event_id['100005']], 
                          event_id={'100005': event_id['100005']},
                          tmin=0, tmax=2, baseline=(0, 0), preload=True)
    
    # 获取放电事件的位置
    dis_events = events[events[:, 2] == event_id['dis']]
    
    return epochs_all, dis_events, events

def create_dataframe(epochs_data, start_epoch, num_epochs, channel_names):
    """创建DataFrame"""
    data_list = []
    for i in range(num_epochs):
        epoch_idx = start_epoch + i
        if epoch_idx >= 0 and epoch_idx < len(epochs_data):
            epoch = epochs_data[epoch_idx]
            # epochs_data的形状为(epochs, channels, times)
            # 我们不需要第一个维度的索引[0]
            time_points = np.linspace(i*2, (i+1)*2, epoch.shape[1])
            
            for t in range(epoch.shape[1]):
                row = {
                    'epochs_id': epoch_idx,
                    'times': time_points[t],
                    'id': t
                }
                for ch_idx, ch_name in enumerate(channel_names):
                    row[ch_name] = epoch[ch_idx, t]  # 删除了[0]索引
                data_list.append(row)
    
    return pd.DataFrame(data_list)

def filter_close_events(dis_events, raw_info, min_epochs_distance=3):
    """过滤掉相距太近的放电事件，保留中间的事件"""
    if len(dis_events) <= 1:
        return dis_events
    
    # 转换事件时间到epoch索引
    epoch_indices = [int(event[0] / raw_info['sfreq'] // 2) for event in dis_events]
    filtered_indices = []
    i = 0
    
    while i < len(epoch_indices):
        # 寻找连续的近距离事件
        j = i + 1
        while j < len(epoch_indices) and \
              (epoch_indices[j] - epoch_indices[j-1]) <= min_epochs_distance:
            j += 1
            
        if j - i > 1:
            # 如果找到多个近距离事件，选择中间的事件
            mid_idx = i + (j - i) // 2
            filtered_indices.append(mid_idx)
        else:
            # 如果是单独的事件，直接保留
            filtered_indices.append(i)
        i = j
    
    return dis_events[filtered_indices]

def process_file(file_path, output_dir):
    """处理单个文件"""
    print(f"Processing {file_path}")
    
    # 读取数据
    raw = load_eeg_data(file_path)
    
    # 获取epochs和事件
    epochs_all, dis_events, events = get_events_and_epochs(raw)
    
    # 过滤相近的放电事件
    filtered_dis_events = filter_close_events(dis_events, raw.info)
    
    # 获取通道名称
    channel_names = raw.ch_names
    
    # 获取数据
    epochs_data = epochs_all.get_data()
    
    # 对于每个放电事件
    for dis_idx, dis_event in enumerate(filtered_dis_events):
        # 找到放电事件所在的epoch
        dis_time = dis_event[0] / raw.info['sfreq']
        dis_epoch = int(dis_time // 2)
        
        # 创建before和after数据
        before_data = create_dataframe(epochs_data, dis_epoch-5, 5, channel_names)
        after_data = create_dataframe(epochs_data, dis_epoch+1, 5, channel_names)
        
        # 构建输出文件名
        base_name = os.path.basename(file_path)
        name_parts = base_name.split('_')
        before_filename = f"{name_parts[0]}_{name_parts[1]}_BeforeTask_GRA_F_Beforedis_{dis_idx}.tsv"
        after_filename = f"{name_parts[0]}_{name_parts[1]}_BeforeTask_GRA_F_Afterdis_{dis_idx}.tsv"
        
        # 保存数据
        before_data.to_csv(os.path.join(output_dir, before_filename), sep='\t', index=False)
        after_data.to_csv(os.path.join(output_dir, after_filename), sep='\t', index=False)
        
        print(f"Saved {before_filename} and {after_filename}")

def main():
    # 设置输入输出路径
    input_dir = "./data/BeforeTask(2016-2020原始数据)/"
    output_dir = "./data/BeforeTask(放电前后数据)/"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有.set文件
    for filename in os.listdir(input_dir):
        if filename.endswith("_BeforeTask_GRA_F.set"):
            file_path = os.path.join(input_dir, filename)
            try:
                process_file(file_path, output_dir)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == '__main__':
    main()