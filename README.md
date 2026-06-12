![Python](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?style=flat-square&logo=pytorch)
![Transformers](https://img.shields.io/badge/Transformers-4.30-orange.svg?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)
![AICUP](https://img.shields.io/badge/AICUP-2026-red.svg?style=flat-square)
# AI CUP 2026 春季賽 基於時序資料之桌球戰術與結果預測競賽

**TEAM**: TEAM_10669  
**Private Leaderboard**: 0.3765799 

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
```

```
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
```

## 模型核心超參數配置 (Core Hyperparameters)

> `src/FinalCode.py`所使用的模型參數

<table>
  <thead>
    <tr style="background-color: #f8f9fa;">
      <th align="left" width="45%">超參數名稱 (Hyperparameter)</th>
      <th align="left" width="15%">配置數值 (Value)</th>
      <th align="left" width="40%">說明 (Description)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>嵌入維度 (Embedding Dimension)</b></td>
      <td><code>32</code></td>
      <td>將輸入標記（Tokens）映射至低維連續向量空間。</td>
    </tr>
    <tr>
      <td><b>隱藏層維度 (Hidden Dimension)</b></td>
      <td><code>256</code></td>
      <td>前饋神經網路與注意力層的內部特徵維度。</td>
    </tr>
    <tr>
      <td><b>注意力標頭數 (Transformer Heads)</b></td>
      <td><code>8</code></td>
      <td>多頭注意力機制並行捕捉不同子空間特徵的數量。</td>
    </tr>
    <tr>
      <td><b>隨機失活率 (Dropout Rate)</b></td>
      <td><code>0.3</code></td>
      <td>用於防止模型過擬合（Overfitting）的正則化比例。</td>
    </tr>
    <tr>
      <td><b>批次大小 (Batch Size)</b></td>
      <td><code>64</code></td>
      <td>每次梯度更新時同時輸入模型的樣本數量。</td>
    </tr>
  </tbody>
</table>
