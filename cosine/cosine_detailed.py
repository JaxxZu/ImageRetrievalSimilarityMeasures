import numpy as np
import os

"""
實現 Cosine Similarity 檢索前10位相似圖片 (仿照 PCC 格式)

參數：
__DATASET_FILES__ ：dataset檔案名
__SEARCH_ITSELF__：檢索結果是否包含自己，True為包含自己，False為不包含自己

"""

__DATASET_FILES__ = [
    "../fullset_original.txt",
    "../fullset_minmax.txt",
    "../fullset_l2.txt",
    "../fullset_zscore.txt",
    "../fullset_zscore_mix_l2.txt"
]

__SEARCH_ITSELF__ = False
__OUTPUT_FOLDER_NAME__ = "retrieval_cosine"  # 改為存到 cosine 資料夾

os.makedirs(__OUTPUT_FOLDER_NAME__, exist_ok=True)


def load_data(filename):
    data = []
    names = []
    classes = []

    print(f"Loading {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        # 跳過第一行標頭 (根據之前的經驗，您的檔案可能有標頭)
        # 如果您的 PCC 代碼能直接跑，代表您的檔案可能沒有標頭，或者第一行被當作數據處理了
        # 為了安全，這裡保留 PCC 的寫法，但請注意如果第一行是字串，float() 會報錯
        # 若報錯，請在此處加入 next(f) 跳過第一行
        
        for i, line in enumerate(f):
            try:
                parts = line.strip().split()
                
                # 根據您的數據格式：特徵...特徵 檔名 類別名
                # parts[:-2] 是特徵
                # parts[-2] 是檔名
                
                if len(parts) < 222: # 簡單檢查行長度
                     continue

                img_name = parts[-2]
                img_class = i // 200 # 依照 PCC 邏輯，每 200 張一類

                # 讀取特徵 (轉為 float)
                features = np.array(list(map(float, parts[:-2])))

                data.append(features)
                names.append(img_name)
                classes.append(img_class)
            except ValueError:
                # 這是為了防範第一行是標頭的情況
                continue

    data = np.array(data)

    # --- Cosine 特有的預處理 ---
    # Cosine 需要計算向量長度 (L2 Norm) 作為分母
    # 預先計算好，之後計算速度會快很多
    data_norm_val = np.linalg.norm(data, axis=1, keepdims=True)
    
    # 避免除以 0
    data_norm_val[data_norm_val == 0] = 1.0 

    return data, names, classes, data_norm_val


def cosine_similarity_fast(vec_a, vec_b, norm_a, norm_b):
    # Cosine = (A . B) / (|A| * |B|)
    dot_product = np.dot(vec_a, vec_b)
    denominator = norm_a * norm_b
    return dot_product / denominator


def search_top10(query_row, data, classes, data_norm_val, f):
    query_vec = data[query_row]
    query_norm = data_norm_val[query_row]
    query_class = classes[query_row]

    scores = []

    # 這裡使用迴圈計算，雖然慢一點但結構與 PCC 代碼一致
    for i in range(len(data)):
        if i == query_row and not __SEARCH_ITSELF__:
            continue

        # 計算 Cosine 分數
        # 這裡傳入單個數值 (item) 而非陣列，所以取 [0]
        sim = cosine_similarity_fast(query_vec, data[i], query_norm[0], data_norm_val[i][0])
        
        # 限制範圍在 -1 到 1 之間 (雖然 Cosine 理論上就是，但浮點數運算可能有誤差)
        sim = max(min(sim, 1.0), -1.0)
        
        scores.append((i, classes[i], sim))

    # 排序：Cosine 越大越相似 (Reverse=True)
    scores.sort(key=lambda x: x[2], reverse=True)
    top10 = scores[:10]

    f.write(f"\n=== Pic:{query_row} ===\n")
    correct = 0

    for row_id, cls, sim in top10:
        if cls == query_class:
            correct += 1
        # 格式對齊 PCC 的輸出格式
        f.write(f"{row_id:<6d}    Cosine={sim:.6f}\n")

    accuracy = correct / 10
    f.write(f"accuracy： {accuracy:.2f}\n============\n")
    return accuracy


if __name__ == "__main__":

    for dataset_file in __DATASET_FILES__:
        
        # 確保檔案存在再執行
        if not os.path.exists(dataset_file):
            print(f"Skipping {dataset_file} (Not Found)")
            continue

        clean_name = os.path.splitext(os.path.basename(dataset_file))[0]
        clean_name = clean_name.replace("fullset_", "")

        # 輸出檔名改為 cosine_xxx.txt
        __OUTPUT_FILE_NAME__ = f"cosine_{clean_name}.txt"
        output_path = os.path.join(__OUTPUT_FOLDER_NAME__, __OUTPUT_FILE_NAME__)

        # 讀取數據 (包含預計算 Norm)
        data, names, classes, data_norm_val = load_data(dataset_file)

        total_accuracy = 0.0

        class_correct = [0] * 50
        class_total = [0] * 50

        print(f"Start processing {dataset_file} -> {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:

            # N = 10000 張圖
            for query_row in range(len(data)):
                if query_row % 100 == 0: # 減少 print 頻率，每 100 張印一次
                    print(f"running on pic: {query_row} / {len(data)}, normalization: {clean_name}")

                acc = search_top10(query_row, data, classes, data_norm_val, f)
                total_accuracy += acc

                cls_id = classes[query_row]  # 0~49

                class_correct[cls_id] += acc
                class_total[cls_id] += 1

            avg_accuracy = total_accuracy / float(len(data))
            f.write("\n===== Advantage Accuracy (Total)=====\n")

            f.write(f"\nadvantage accuracy: {avg_accuracy:.4f}\n")

            f.write("\n===== Class Accuracy =====\n")
            for c in range(50):
                if class_total[c] > 0:
                    class_acc = class_correct[c] / class_total[c]
                else:
                    class_acc = 0.0
                f.write(f"Class {c:02d}:  {class_acc:.4f}\n")
                
        print(f"Finished {clean_name}. Saved to {output_path}\n")