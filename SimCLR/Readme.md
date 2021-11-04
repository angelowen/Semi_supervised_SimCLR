# Semi-supervised lerning - Simclr

## SimLR 論文介紹
https://bbs.cvmart.net/articles/4950

## Dataset
STL-10 是一個圖像數據集，包含 10 類物體的圖片，每類 1300 張圖片，500 張訓練，800 張測試，每張圖片分辨率為 96x96。除了具有類別標籤的圖片之外，還有 100000 張Unlabel的圖片。

## command
1. `python run.py`
2. find the path of saving model
3. `python finetune.py`
4. `test.py`