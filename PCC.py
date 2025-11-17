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

    data_mean = data.mean(axis=1, keepdims=True)
    data_std = data.std(axis=1, keepdims=True)
    data_std[data_std == 0] = 1
    data_norm = (data - data_mean) / data_std

    return data, names, classes, data_norm



def pcc_similarity_fast(norm_a, norm_b):
    return np.dot(norm_a, norm_b)


def search_top10(query_row, data, classes, data_norm, f):
    query_vec_norm = data_norm[query_row]
    query_class = classes[query_row]

    scores = []

    for i in range(len(data)):
        if i == query_row and not __SEARCH_ITSELF__:
            continue

        sim = pcc_similarity_fast(query_vec_norm, data_norm[i]) / data.shape[1]
        sim = max(min(sim, 1.0), -1.0)
        scores.append((i, classes[i], sim))

    scores.sort(key=lambda x: x[2], reverse=True)
    top10 = scores[:10]

    f.write(f"\n=== Pic:{query_row} ===\n")
    correct = 0

    for row_id, cls, sim in top10:
        if cls == query_class:
            correct += 1
        f.write(f"{row_id:<6d}    PCC={sim:.6f}\n")

    accuracy = correct / 10
    f.write(f"accuracy： {accuracy:.2f}\n============\n")
    return accuracy



if __name__ == "__main__":

    for dataset_file in __DATASET_FILES__:

        clean_name = os.path.splitext(os.path.basename(dataset_file))[0]
        clean_name = clean_name.replace("fullset_", "")

        __OUTPUT_FILE_NAME__ = f"pcc_{clean_name}.txt"
        output_path = os.path.join(__OUTPUT_FOLDER_NAME__, __OUTPUT_FILE_NAME__)

        data, names, classes, data_norm = load_data(dataset_file)

        total_accuracy = 0.0

        class_correct = [0] * 50
        class_total = [0] * 50

        with open(output_path, "w", encoding="utf-8") as f:

            for query_row in range(10000):
                print("running on pic:", query_row, ", normalization:", clean_name)

                acc = search_top10(query_row, data, classes, data_norm, f)
                total_accuracy += acc

                cls_id = classes[query_row]  # 0~49

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
                    class_acc = 0.0
                f.write(f"Class {c:02d}:  {class_acc:.4f}\n")
