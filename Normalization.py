import numpy as np


"""
讀取fulldataset進行正規化
"""

input_file = "fullset_original.txt"

data = []
filenames = []
labels = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        feats = list(map(float, parts[:-2]))   # 特徵
        filename = parts[-2]                  # 圖檔名
        label = parts[-1]                     # 類別
        data.append(feats)
        filenames.append(filename)
        labels.append(label)

data = np.array(data)


#  Z-score Normalization

mean = data.mean(axis=0)
std = data.std(axis=0)
std[std == 0] = 1 # 避免除0

zscore_data = (data - mean) / std


# L2 Normalization

l2_norm = np.linalg.norm(data, axis=1, keepdims=True)
l2_norm[l2_norm == 0] = 1
l2_data = data / l2_norm


#  Min–Max Normalization

d_min = data.min(axis=0)
d_max = data.max(axis=0)
range_val = d_max - d_min
range_val[range_val == 0] = 1

minmax_data = (data - d_min) / range_val


# Z-score + L2 Normalization

z_l2_norm = np.linalg.norm(zscore_data, axis=1, keepdims=True)
z_l2_norm[z_l2_norm == 0] = 1
zscore_l2_data = zscore_data / z_l2_norm



def save_file(filename_out, norm_data):
    with open(filename_out, "w", encoding="utf-8") as f:
        for i in range(len(norm_data)):
            vals = " ".join(map(str, norm_data[i]))
            f.write(f"{vals} {filenames[i]} {labels[i]}\n")


save_file("fullset_zscore.txt", zscore_data)
save_file("fullset_l2.txt", l2_data)
save_file("fullset_minmax.txt", minmax_data)
save_file("fullset_zscore_mix_l2.txt", zscore_l2_data)

print("all done")
