import streamlit as st
from PIL import Image
import os

Basic_dir = r"電腦視覺專案圖片\pic\\"

img_folder = ["AncestorDinoArt", "Archit", "Baseball","Basketball", "Beach", "Billiardsball", "Bus"
          ,"BWimage","Car","Cartoon","Castle","Citynight","ClassicalPainting","Cropcycle", "DeerAntelope"
          ,"Desert","Dog","Doors","Eagle","Elephant","F1","Feasts","Flower","Grass","Group","Indoor","Lion"
          ,"Masks","Model","Mountain","Owls","Penguin","Plane","Planet","Pumpkin","RacingMotor","Satelliteimage"
          ,"Sculpt","Ship","Sky","Soccer","Stalactite","SubSea","Sunflower","Sunset","Surfs","Tennis","Tiger"
          ,"Volleyball","Waterfall"]

img_list = []

def parse_pic_blocks(txt_path):
    blocks = {}
    current_pic = None
    current_lines = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 開始一個 Pic 區塊，例如: === Pic:10 ===
            if line.startswith("=== Pic:"):
                if current_pic is not None:
                    blocks[current_pic] = parse_pic_content(current_lines)

                # 抓 Pic 編號
                current_pic = int(line.replace("=== Pic:", "").replace("===", "").strip())
                current_lines = []

            # 區塊結束（行是 =======）
            elif line.startswith("==="):
                if current_pic is not None:
                    blocks[current_pic] = parse_pic_content(current_lines)
                current_pic = None
                current_lines = []

            else:
                if current_pic is not None:
                    current_lines.append(line)

    return blocks


def parse_pic_content(lines):
    items = []
    accuracy = None

    for line in lines:
        line = line.strip()
        # 有 "accuracy" 的那行
        if "accuracy" in line:
            # 假設格式: accuracy： 0.90
            accuracy = float(line.replace("accuracy：", "").strip())
            continue

        # 正常資料行，例如:
        # 98        PCC=0.997836
        if "PCC=" in line:
            parts = line.split()
            id_num = int(parts[0])         # 前面的 id
            pcc_val = float(parts[-1].split("=")[1])  # PCC=後面的值
            items.append({"id": id_num, "score": pcc_val})
        elif "Euc=" in line:
            parts = line.split()
            id_num = int(parts[0])         # 前面的 id
            pcc_val = float(parts[-1].split("=")[1])  # PCC=後面的值
            items.append({"id": id_num, "score": pcc_val})
        elif "Cosine=" in line:
            parts = line.split()
            id_num = int(parts[0])         # 前面的 id
            pcc_val = float(parts[-1].split("=")[1])  # PCC=後面的值
            items.append({"id": id_num, "score": pcc_val})

    return {
        "items": items,
        "accuracy": accuracy
    }


# ---- 基本設定 ----
st.set_page_config(
    page_title="電腦視覺期中報告",
    # page_icon="📊",
    layout="wide"
)

def show_home(cal_option, option, selected_folder):
    st.title("電腦視覺期中報告")
    st.write("使用說明：  \n⬅️於左側sidebar選擇相似度計算方式、正規化模式以及圖片類別  \n⬇️於下方左側選擇需要查看的圖片，結果將顯示於下方右側↘️")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("選擇要查看的圖片")
        # cal_option = st.selectbox(
        #     "請選擇相似度模式：",
        #     ["euclidean 歐基里德", "PCC","cosine"]
        # )
        # option = st.selectbox(
        #     "請選擇正規化模式：",
        #     ["無", "L2正規化", "Zscore正規化","L2+Zscore正規化","MinMax正規化"]
        # )

        selected = None
        # st.write("點圖片按鈕以選擇：")
        
        # selected_folder = st.selectbox("選擇圖片資料夾：", img_folder)
        if "page" not in st.session_state:
            st.session_state.page = 1

        images_per_page = 20
        total_pages = (200 - 1) // images_per_page + 1

        col_prev, col_info, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.button("⬅ 上一頁") and st.session_state.page > 1:
                st.session_state.page -= 1

        with col_next:
            if st.button("下一頁 ➡") and st.session_state.page < total_pages:
                st.session_state.page += 1

        with col_info:
            st.write(f"第 {st.session_state.page} / {total_pages} 頁")
        page = st.session_state.page
        start_idx = (page - 1) * images_per_page
        end_idx = start_idx + images_per_page
        page_imgs = img_list[img_folder.index(selected_folder)][start_idx:end_idx]
        
        
        cols = st.columns(3)
        # {Basic_dir}{img_folder[img_folder.index(selected_folder)]}/{img_name}
        # if selected_folder
        for i, img_name in enumerate(page_imgs):
            # img_path = os.path.join(folder_path, img_name)
            img = Image.open(f"{Basic_dir}{img_folder[img_folder.index(selected_folder)]}/{img_name}")

            with cols[i % 3]:
                st.image(img, caption=img_name, use_container_width=True)

                if st.button(f"選擇 {img_name}"):
                    selected = img_name

        
            # st.image(f"{img_folder[img_folder.index(selected_folder)]}/{selected}")

    with col2:
        st.subheader("相似圖片：")
        if selected and option and cal_option:
            st.success(f"你選擇了：{selected}  \n正規化方式：{option}  \n距離計算公式：{cal_option}")
            img = Image.open(f"{Basic_dir}{img_folder[img_folder.index(selected_folder)]}/{selected}")
            st.image(img, caption=selected, use_container_width=True)
            pic_number = img_list[img_folder.index(selected_folder)].index(selected) + img_folder.index(selected_folder)*200

            if option == "無":
                option = "original"
            elif option == 'L2正規化':
                option = "l2"
            elif option == "Zscore正規化":
                option = "zscore"
            elif option == "L2+Zscore正規化":
                option = "zscore_mix_l2"
            elif option == "MinMax正規化":
                option = "minmax"

            if cal_option == "euclidean 歐基里德":
                data = parse_pic_blocks(f"retrieval_euclidean/euclidean_{option}.txt")
            elif cal_option == "PCC":
                data = parse_pic_blocks(f"retrieval_pcc/pcc_{option}.txt")
            elif cal_option == "cosine":
                data = parse_pic_blocks(f"cosine/retrieval_cosine/cosine_{option}.txt")
            
            st.success(f"ACC: {data[pic_number]['accuracy']}")
            cols = st.columns(3)
            for i, item in enumerate(data[pic_number]["items"]):
                target_folder = int(item['id'] /200)
                target_img = str(int(item['id']%200)).zfill(3)
                img = Image.open(f"{Basic_dir}{img_folder[target_folder]}/{img_folder[target_folder]}_{target_img}.jpg")

                with cols[i % 3]:
                    st.image(img, caption=f"{img_folder[target_folder]}_{target_img}.jpg\n{option} Score ： {item['score']}", use_container_width=True)


def main():
    # # ---- Sidebar 導覽列 ----
    st.sidebar.title("🔧 操作選單")
    cal_option = st.sidebar.selectbox(
        "請選擇相似度模式：",
        ["euclidean 歐基里德", "PCC","cosine"]
    )
    option = st.sidebar.selectbox(
        "請選擇正規化模式：",
        ["無", "L2正規化", "Zscore正規化","L2+Zscore正規化","MinMax正規化"]
    )
    selected_folder = st.sidebar.selectbox("選擇圖片資料夾：", img_folder)

    show_home(cal_option=cal_option, option=option, selected_folder=selected_folder)



if __name__ == "__main__":
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
    for img_dir in img_folder:
        img_list.append([img for img in os.listdir(Basic_dir+img_dir) if img.lower().endswith(valid_ext)])
    main()
