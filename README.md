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

# 基于时序数据之桌球战术与结果预测竞赛 🚀

![Python](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?style=flat-square&logo=pytorch)
![Transformers](https://img.shields.io/badge/Transformers-4.30-orange.svg?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)
![AICUP](https://img.shields.io/badge/AICUP-2026-red.svg?style=flat-square)

本专案为 **AICUP 2026 基于时序数据之桌球战术与结果预测竞赛** 的官方开源代码库。
团队核心创新在于针对桌球/羽球赛事的「强时序依赖性」与「长尾分布」，设计了一套结合 **单向 LSTM** 与 **Transformer 编码器** 的双层混合多任务时序专家系统（MultiTaskLSTMTransformer），在盲测集上展现出极高的强健性与泛化能力。

---

## 📁 专案目录结构

```text
project_root/
│
├── data/                  # 竞赛原始资料集
│   ├── train.csv          # 训练集数据
│   ├── test_new.csv       # 盲测集数据
│   └── sample_submission.csv
│
├── src/                   # 核心核心程式码
│   ├── baseline code.py   # 基础 Baseline 模型
│   └── FinalCode.py       # 最终采用的混合时序多任务模型 (v4.2)
│
├── checkpoints/           # 5 折交叉验证中所储存之最佳模型权重点 (.pt)
│   ├── fold0_best.pt, fold1_best.pt, ...
│   └── best_lstm_model.pt
│
└── submissions/           # 模型推论后输出之最终预测结果 CSV 档案
    └── submission_hybrid_v4_2.csv

[ 9项类别特征输入 ] (性别、持拍手、力量、旋转、球种、击球动作、击球位置、击球落点、击球拍数)
        │
        ▼
[ 特征嵌入层 (Embedding) ] ➔ 9组独立 Embedding Matrix (各32维)
        │
        ▼
[ 特征拼接 (Concat) ] ➔ 融合成 288 维之复合语义时序向量
        │
        ▼
[ 单向 LSTM 网络 ] ➔ 隐藏层 256 维，利用门控机制平滑物理轨迹动态杂讯
        │
        ▼
[ 位置编码 (Positional) ] ➔ 注入绝对时序位置资讯
        │
        ▼
[ Transformer 编码器 ] ➔ 多头注意力机制 (8 Heads, Dropout 0.3)
   ⚠️ [下三角因果遮罩] ➔ 动态将 t+1 拍后之权重归零，严格杜绝「偷看未来」资料洩漏！
        │
        ▼
[ 特征解耦多任务输出头 (Multi-Task Decoupling Heads) ]
        ├── 🎯 动作分类头 (act_head) ➔ 预测下一拍动作 (Balanced Focal Loss)
        ├── 📍 落點分类头 (pt_head)  ➔ 预测下一拍落点 (Balanced Focal Loss)
        └── 🏆 胜负预测头 (rly_head)  ➔ 特征 Mean Pooling ➔ 预测回合胜负 (BCE Loss)
