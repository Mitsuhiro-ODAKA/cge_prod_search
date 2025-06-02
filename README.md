# Automatic CGE Model Construction from Causal Rules and Observational Data

**観測データと背景知識に基づく応用一般均衡モデル生産サイド（縮約版成長回帰モデル）構造の自動生成**

## Overview

本プロジェクトは、**観測データと背景知識に基づいて応用一般均衡モデル（CGE）の生産関数を構造的に自動生成**するPythonベースの実験的ワークフローです。

- ブーリアン因果ルールの抽出（データ駆動＋LLM補助）
- 矛盾ルールの除去（ASP: Answer Set Programming）
- 回帰による係数推定（sklearn）
- Cobb-Douglas型生産関数の推定
- GAMSpyによる最適化モデル構築・解法

## Application Context

- 世界銀行WDIからのデータ取得（GDP, CO₂, 再エネ, 労働力, 資本形成 など）
- モデル構造の変化を通じて因果シナリオの比較分析が可能

## Project Structure
```
├── data/
│ └── worldbank_data.csv # 取得済みのWDIデータ
├── rules/
│ └── cleaned_data_rules.txt # 選別済みブーリアンルール
├── models/
│ └── estimated_model.gms # 自動生成されたGAMSコード
├── notebooks/
│ └── causal_to_cge_workflow.ipynb # モデル構築と可視化の全過程
├── src/
│ └── build_cge_model.py # コアの自動生成ロジック
├── config.yaml # 国・指標・期間などの設定ファイル
└── README.md # 本ファイル
```

## 🔧 Requirements

- Python >= 3.8
- `gamspy`, `scikit-learn`, `pandas`, `numpy`, `wbdata`, `networkx`, `clingo`, `pyyaml`
- GAMS (for solving the model)

```bash
pip install -r requirements.txt
```

![01](imgs/01.png)