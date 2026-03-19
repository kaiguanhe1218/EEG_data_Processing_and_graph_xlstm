import os
import mne
import numpy as np
import pandas as pd
import random

def load_eeg_data(file_path):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    if raw.times[-1] > 500:
        raw.crop(tmin=0, tmax=500)
    return raw

def get_events_and_epochs(raw):
    events, event_id = mne.events_from_annotations(raw)
    # 只处理 '100005' 标签，'dis' 可选
    if '100005' not in event_id:
        raise ValueError("event_id 不包含 '100005'，无法分 epoch")
    epochs_all = mne.Epochs(raw, events=events[events[:, 2] == event_id['100005']],
                          event_id={'100005': event_id['100005']},
                          tmin=0, tmax=2, baseline=(0, 0), preload=True)
    if 'dis' in event_id:
        dis_events = events[events[:, 2] == event_id['dis']]
    else:
        dis_events = np.array([])
    return epochs_all, dis_events, events

def create_dataframe(epochs_data, start_epoch, num_epochs, channel_names):
    data_list = []
    for i in range(num_epochs):
        epoch_idx = start_epoch + i
        if epoch_idx >= 0 and epoch_idx < len(epochs_data):
            epoch = epochs_data[epoch_idx]
            time_points = np.linspace(i*2, (i+1)*2, epoch.shape[1])
            for t in range(epoch.shape[1]):
                row = {
                    'epochs_id': epoch_idx,
                    'times': time_points[t],
                    'id': t
                }
                for ch_idx, ch_name in enumerate(channel_names):
                    row[ch_name] = epoch[ch_idx, t]
                data_list.append(row)
    return pd.DataFrame(data_list)

def process_no_discharge_file(file_path, output_dir_nodis, window_size=5, num_segments=4):
    print(f"Processing no-discharge file: {file_path}")
    raw = load_eeg_data(file_path)
    epochs_all, dis_events, events = get_events_and_epochs(raw)
    channel_names = raw.ch_names
    epochs_data = epochs_all.get_data()
    total_epochs = len(epochs_data)
    # 如果没有放电事件
    if len(dis_events) == 0 and total_epochs > window_size:
        # 随机选取4个区间
        possible_starts = list(range(0, total_epochs - window_size))
        if len(possible_starts) < num_segments:
            chosen_starts = possible_starts
        else:
            chosen_starts = random.sample(possible_starts, num_segments)
        base_name = os.path.basename(file_path)
        name_parts = base_name.split('_')
        for idx, start_epoch in enumerate(chosen_starts):
            inter_data = create_dataframe(epochs_data, start_epoch, window_size, channel_names)
            inter_filename = f"{name_parts[0]}_{name_parts[1]}_BeforeTask_GRA_F_Nodis_{idx}.tsv"
            inter_data.to_csv(os.path.join(output_dir_nodis, inter_filename), sep='\t', index=False)
            print(f"Saved {inter_filename}")

def main():
    input_dir = "./data/BeforeTask(2016-2020原始数据)/"
    output_dir_nodis = "./data/BeforeTask(未放电)/"
    os.makedirs(output_dir_nodis, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith("_BeforeTask_GRA_F.set"):
            file_path = os.path.join(input_dir, filename)
            try:
                raw = load_eeg_data(file_path)
                try:
                    epochs_all, dis_events, _ = get_events_and_epochs(raw)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
                if dis_events is None or len(dis_events) == 0:
                    process_no_discharge_file(file_path, output_dir_nodis)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == '__main__':
    main()