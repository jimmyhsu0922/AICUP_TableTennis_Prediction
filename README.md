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
