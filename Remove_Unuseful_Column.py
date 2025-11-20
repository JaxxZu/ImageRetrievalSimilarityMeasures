import os
""" 
This script removes unuseful columns from raw data
"""

input_folder = r"raw_data"
output_folder = r"removed_unuseful_column_data"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".txt"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            # remove first row
            lines = f.readlines()[1:]

        processed_lines = []

        for line in lines:
            # remove including Thumbs.db's row
            if "Thumbs.db" in line:
                continue

            cols = line.strip().split()

            # remove scalable's col
            for idx in range(335, 79, -1):
                if idx < len(cols):
                    cols.pop(idx)

            # remove path and counter
            if len(cols) >= 4:
                cols.pop(-2)
                cols.pop(-3)

            processed_lines.append(" ".join(cols) + "\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(processed_lines)

print("all done")
