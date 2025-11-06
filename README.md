# ImageRetrievalSimilarityMeasures
影像檢索中的相似度量測比較

## 題目要求
以每一個類別一檔案，一個類別有200張圖  
50個類別，共10000張圖
每圖有478個欄位  
|欄位|內容|
|:----:|:---:|
|0-31欄| ColorStructure|
|32-43欄| ColLayout|
|44-79欄| RegionShape|
|80-335欄| ScalableColor|
|336-397欄| HomogeneousTexture|
|398-477欄| EdgeHistogram|  

請將檔案分成正規化(以欄為單位)與非正規化兩種 
讀取特徵檔案，用每一張圖當Query Image，去搜尋fnormal.txt或unformal.txt，取最像的前10名，結果不能含Query自己，計算準確率  
從多角度去比較與分析 Cosine Similarity,Euclidean Distance與PCC的準確率  
加分: 實作成一系統  
每類圖片隨機挑選一張當Query  
使用者可選擇是否正規化，與相似度比對方法，送出查詢  
系統會執行比對程式，挑出最像10名，回傳  

## 成員分工
|成員|分工|
|:----:|:---:|
| @Rex0626|Cosine Similarity部分 |
| @Cecilia1050300 |Euclidean Distance部分 |
| @JaxxZu |PCC部分 |
| @yieie |檢索系統 |

