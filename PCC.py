import numpy as np
import os

"""
實現 PCC 檢索前10位相似圖片

參數：
__DATASET_FILES__ ：dataset檔案名
__SEARCH_ITSELF__：檢索結果是否包含自己，True為包含自己，False為不包含自己
"""

__DATASET_FILES__ = [
    "fullset_original.txt",
    "fullset_minmax.txt",
    "fullset_l2.txt",
    "fullset_zscore.txt",
    "fullset_zscore_mix_l2.txt"
]

__SEARCH_ITSELF__ = False
__OUTPUT_FOLDER_NAME__ = "retrieval_pcc"

os.makedirs(__OUTPUT_FOLDER_NAME__, exist_ok=True)


def load_data(filename):
    data = []
    names = []
    classes = []

    print(f"Loading {filename} ...")
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()

            img_name = parts[-2]
            img_class = i // 200

            features = np.array(list(map(float, parts[:-2])))

            data.append(features)
            names.append(img_name)
            classes.append(img_class)

    data = np.array(data)

    # 預處理 PCC 需要的標準化 (Z-score row-wise)
    data_mean = data.mean(axis=1, keepdims=True)
    data_std = data.std(axis=1, keepdims=True)
    data_std[data_std == 0] = 1
    data_norm = (data - data_mean) / data_std

    return data, names, classes, data_norm


def search_top10(query_row, data, classes, data_norm, f):
    query_vec_norm = data_norm[query_row]
    query_class = classes[query_row]


    # 1. 向量化計算：一次算出 Query 與所有圖片的點積 (N,)
    sims = np.dot(data_norm, query_vec_norm) / data.shape[1]

    # 2. 數值修剪 (Clip)，防止浮點數誤差超出 [-1, 1]
    sims = np.clip(sims, -1.0, 1.0)

    # 3. 處理不包含自己的邏輯
    if not __SEARCH_ITSELF__:
        # 將自己的分數設為極小值 (-2.0)，排序時自然會沉底
        sims[query_row] = -999.0

    # 4. 取最大的 10 個
    top10_indices = np.argpartition(sims, -10)[-10:]
    # 對這 10 個再由大到小排序
    top10_indices = top10_indices[np.argsort(sims[top10_indices])[::-1]]


    f.write(f"\n=== Pic:{query_row} ===\n")
    correct = 0

    for idx in top10_indices:
        sim = sims[idx]
        cls = classes[idx]

        if cls == query_class:
            correct += 1

        f.write(f"{idx:<6d}    PCC={sim:.6f}\n")

    accuracy = correct / 10
    f.write(f"accuracy： {accuracy:.2f}\n============\n")
    return accuracy


if __name__ == "__main__":

    for dataset_file in __DATASET_FILES__:
        if not os.path.exists(dataset_file):
            print(f"Skipping {dataset_file} (Not Found)")
            continue

        clean_name = os.path.splitext(os.path.basename(dataset_file))[0]
        clean_name = clean_name.replace("fullset_", "")

        __OUTPUT_FILE_NAME__ = f"pcc_{clean_name}.txt"
        output_path = os.path.join(__OUTPUT_FOLDER_NAME__, __OUTPUT_FILE_NAME__)

        data, names, classes, data_norm = load_data(dataset_file)

        total_accuracy = 0.0

        class_correct = [0] * 50
        class_total = [0] * 50

        print(f"Start processing {clean_name} ...")

        with open(output_path, "w", encoding="utf-8") as f:

            for query_row in range(10000):  # 保留原本 Hardcode

                # 每 1000 張印一次進度
                if query_row % 1000 == 0:
                    print(f"running on pic: {query_row}, normalization: {clean_name}")

                acc = search_top10(query_row, data, classes, data_norm, f)
                total_accuracy += acc

                cls_id = classes[query_row]

                if 0 <= cls_id < 50:
                    class_correct[cls_id] += acc
                    class_total[cls_id] += 1

            avg_accuracy = total_accuracy / 10000.0
            f.write("\n===== Advantage Accuracy (Total)=====\n")

            f.write(f"\nadvantage accuracy: {avg_accuracy:.4f}\n")

            f.write("\n===== Class Accuracy =====\n")
            for c in range(50):
                if class_total[c] > 0:
                    class_acc = class_correct[c] / class_total[c]
                else:
                    class_acc = -1.0
                f.write(f"Class {c:02d}:  {class_acc:.4f}\n")

        print(f"Finished {clean_name} \n")