# ImageRetrievalSimilarityMeasures
影像檢索中的相似度量測比較

## 題目要求
以每一個類別一檔案，一個類別有200張圖  
50個類別，共10000張圖
每圖有478個欄位  
|欄位|內容|備註|
|:----:|:---:|:---:|
|0-31欄| ColorStructure| |
|32-43欄| ColLayout| |
|44-79欄| RegionShape| |
|~80-335欄~| ~ScalableColor~|原data檔為全0，無需理會這些欄位|
|336-397欄| HomogeneousTexture| |
|398-477欄| EdgeHistogram| |  

請將檔案分成正規化(以欄為單位)與非正規化兩種 
讀取特徵檔案，用每一張圖當Query Image，去搜尋fnormal.txt或unformal.txt，取最像的前10名，結果不能含Query自己，計算準確率  
從多角度去比較與分析 Cosine Similarity,Euclidean Distance與PCC的準確率  
加分: 實作成一系統  
每類圖片隨機挑選一張當Query  
使用者可選擇是否正規化，與相似度比對方法，送出查詢  
系統會執行比對程式，挑出最像10名，回傳  

## 正規化
<img width="600px" src="readme/Normalization.svg"/>  

### Remove_Unuseful_Column.py
移除原dataset中無用的欄位，僅保留特徵、檔案名、類別名，分別保存到`removed_unuseful_column_data`資料夾中。  
### Mix_Dataset.py  
將50個dataset根據檔案名a~z合併成一個完整dataset `fullset_original.txt`。  
### Normalization.py  
讀取完整dataset進行正規化，產生`fullset_zscore.txt`、`fullset_l2.txt`、`fullset_minmax.txt`、`fullset_zscore_mix_l2.txt`。  

## 圖像檢索結果
### PCC.py  
對`fullset_original.txt`、`fullset_zscore.txt`、`fullset_l2.txt`、`fullset_minmax.txt`、`fullset_zscore_mix_l2.txt`進行每張圖片檢索，結果保存在`retrieval_pcc`資料夾中。  
### Euclidean_Distance.py
對`fullset_original.txt`、`fullset_zscore.txt`、`fullset_l2.txt`、`fullset_minmax.txt`、`fullset_zscore_mix_l2.txt`進行每張圖片檢索，結果保存在`retrieval_euclidean`資料夾中。  



## 檢索系統
working


## 成員分工
|成員|分工|
|:----:|:---:|
| [@JaxxZu](https://github.com/JaxxZu) |移除無用欄位、正規化處理dataset、實作PCC部分 |
| [@Rex0626](https://github.com/Rex0626)|實作Cosine Similarity部分 |
| [@Cecilia1050300](https://github.com/Cecilia1050300) |實作Euclidean Distance部分 |
| [@yieie](https://github.com/yieie) |檢索系統 |

