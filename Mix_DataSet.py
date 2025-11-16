import os

folder = "removed_unuseful_column_data"
output_file = "fulldataset.txt"

files = [f for f in os.listdir(folder) if f.endswith(".txt")]
files.sort()   # 依照檔名字母順序排序

with open(output_file, "w", encoding="utf-8") as out:
    for name in files:
        path = os.path.join(folder, name)
        with open(path, "r", encoding="utf-8") as f:
            out.write(f.read())
