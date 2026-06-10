![Python](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?style=flat-square&logo=pytorch)
![Transformers](https://img.shields.io/badge/Transformers-4.30-orange.svg?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)
![AICUP](https://img.shields.io/badge/AICUP-2026-red.svg?style=flat-square)
# 基於時序資料之桌球戰術與結果預測競賽 

這是我們在 AICUP 競賽中的官方開源程式碼庫。本專案建構於 Google Colab T4 GPU 環境下，結合了單向 LSTM 與 Transformer 編碼器架構進行多任務（動作、落點、勝負）之時序預測。

## 專案目錄結構

* `src/`：包含基礎 Baseline 模型以及本團隊最終採用的混合時序多任務模型 (`FinalCode.py`)。
* `data/`：競賽原始訓練集與測試集數據。
* `checkpoints/`：5 折交叉驗證中所儲存之最佳模型權重檔 (`.pt` / `.pth`)。
* `submissions/`：模型推論後輸出之最終預測結果 CSV 檔案。

## 環境配置與安裝說明

請確保您的執行環境具備 Python 3.10+ 以及支援 CUDA 的硬體加速：

```bash
# 安裝所需的核心深度學習與數據處理套件
pip install torch pandas numpy scikit-learn transformers

# 基於時序資料之桌球戰術與結果預測競賽



本專案為 **AICUP 2026 基於時序資料之桌球戰術與結果預測競賽** 的官方開源程式碼庫。
團隊核心創新在於針對桌球/羽球賽事的「強時序依賴性」與「長尾分佈」，設計了一套結合 **單向 LSTM** 與 **Transformer 編碼器** 的雙層混合多任務時序專家系統（MultiTaskLSTMTransformer），在盲測集上展現出極高的強健性與泛化能力。

---

## 專案目錄結構

project_root/
│
├── data/                  # 競賽原始資料集
│   ├── train.csv          # 訓練集數據
│   ├── test_new.csv       # 盲測集數據
│   └── sample_submission.csv
│
├── src/                   # 核心核心程式碼
│   ├── baseline code.py   # 基礎 Baseline 模型
│   └── FinalCode.py       # 最終採用的混合時序多任務模型 (v4.2)
│
├── checkpoints/           # 5 折交叉驗證中所儲存之最佳模型權重點 (.pt)
│   ├── fold0_best.pt, fold1_best.pt, ...
│   └── best_lstm_model.pt
│
└── submissions/           # 模型推論後輸出之最終預測結果 CSV 檔案
    └── submission_hybrid_v4_2.csv


[ 9項類別特徵輸入 ] (性別、持拍手、力量、旋轉、球種、擊球動作、擊球位置、擊球落點、擊球拍數)
        │
        ▼
[ 特徵嵌入層 (Embedding) ] ➔ 9組獨立 Embedding Matrix (各32維)
        │
        ▼
[ 特徵拼接 (Concat) ] ➔ 融合成 288 維之複合語意時序向量
        │
        ▼
[ 單向 LSTM 網路 ] ➔ 隱藏層 256 維，利用門控機制平滑物理軌跡動態雜訊
        │
        ▼
[ 位置編碼 (Positional) ] ➔ 注入絕對時序位置資訊
        │
        ▼
[ Transformer 編碼器 ] ➔ 多頭注意力機制 (8 Heads, Dropout 0.3)
    [下三角因果遮罩] ➔ 動態將 t+1 拍後之權重歸零，嚴格杜絕「偷看未來」資料洩漏！
        │
        ▼
[ 特徵解耦多任務輸出頭 (Multi-Task Decoupling Heads) ]
        ├──  動作分類頭 (act_head) ➔ 預測下一拍動作 (Balanced Focal Loss)
        ├──  落點分類頭 (pt_head)  ➔ 預測下一拍落點 (Balanced Focal Loss)
        └──  勝負預測頭 (rly_head)  ➔ 特徵 Mean Pooling ➔ 預測回合勝負 (BCE Loss)
