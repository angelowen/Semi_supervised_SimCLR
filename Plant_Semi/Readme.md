# Semi-supervised lerning - Kaggle plants
## 參考論文SimLR 介紹
https://bbs.cvmart.net/articles/4950

## Dataset
link: [https://drive.google.com/file/d/1GJEvPH5vuQUq0Sj4b09tqsVNk10uSalq/view?usp=sharing]
比賽數據集，
* 4500 張Unlabled圖片
* 600 張labeled圖片，包含 12 類物體的圖片，每類50張
* 測試資料輸出sample_submission.csv
```
-/datasets/
    /-test/
    /-train/
        /-class1/
            - *.jpg
            ...
        /-class2/
            - *.jpg
            ...
    /-unlabel
        -| 1.jpg
        -| 2.jpg
        ...
    
```
## Command
1. `python train_unlabel.py`
2. find the path of saving model [runs/...] and revise it in `finetune_label.py`
3. `python finetune_label.py`
4. `python test.py`

## Results
* epochs=600, arch=Resnet50 ,batch_size = 32 => Submission Result = 0.80
