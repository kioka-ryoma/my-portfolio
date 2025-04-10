# 競馬レース予測モデル - 自主制作ポートフォリオ

本リポジトリは、自分自身の学習および実験目的のために自主的に開発したプロジェクトです。
競馬レースにおける出走馬の情報をもとに、上位入着を予測する機械学習モデルの構築と推論を目的としています。

個人でデータ収集から前処理、モデル構築、推論、評価まで一貫して実装したものであり、
将来的な業務や研究に活かすためのポートフォリオとして位置付けています。

---

## 📁 構成概要

このプロジェクトは以下の6つのPythonスクリプトで構成されています。

| ファイル名 | 説明 |
|-----------|------|
| `2_extract_race_ids.py` | 過去レースのIDを取得し、学習用に保存するプログラム |
| `1_scrape_race_data.py` | 上記IDをもとにレースデータ（馬名・着順・騎手など）をスクレイピングするプログラム |
| `3_add_pedigree_info.py` | スクレイピング済みデータに馬の血統・通算成績を付与するプログラム |
| `4_train_race_model.py` | 学習済みデータを用いてLightGBMモデルの訓練・アンサンブルを行うプログラム |
| `5_scrape_prediction_race.py` | 予測したい今後のレース情報を取得するプログラム（出走馬・騎手・血統など） |
| `6_predict_race_results.py` | 学習済みモデルと新レースデータを用いて予測と推奨馬を出力するプログラム |

---

## 🔧 使用技術

- 言語: Python 3.10+
- ライブラリ: pandas, numpy, scikit-learn, lightgbm, selenium, BeautifulSoup, re, tkinter など
- モデル: LightGBM による二値分類（上位3着以内予測）
- アンサンブル: 誤分類サンプルへの反復学習と重み付きアンサンブル予測

---

## 📌 使い方（概要）

### 1. 学習データ作成
```bash
python 2_extract_race_ids.py
python 1_scrape_race_data.py
python 3_add_pedigree_info.py
```

### 2. モデル訓練
```bash
python 4_train_race_model.py
```

### 3. 予測用レース情報取得
```bash
python 5_scrape_prediction_race.py
```

### 4. 推論実行
```bash
python 6_predict_race_results.py
```

---

## 📈 評価指標

- F1スコア
- ROC-AUC
- Recall（再現率）
- Confusion Matrix

---

## 🔮 出力例

```
推奨馬（Top 3）:
馬名: ○○, 騎手: △△, 標準化スコア: 1.320, 予測確率: 85.23%
馬名: ××, 騎手: □□, 標準化スコア: 1.021, 予測確率: 79.45%
馬名: ▲▲, 騎手: ☆☆, 標準化スコア: 0.823, 予測確率: 75.12%
```

---

## ⚠️ 注意事項

- 本プロジェクトはあくまで**個人利用・学習用**に作成したものであり、商用・ギャンブル目的での利用は推奨しておりません。
- データは [netkeiba.com](https://db.netkeiba.com/) より取得しており、利用の際は対象サイトの利用規約をご確認ください。

---
## 🔄 今後の実装予定

- 3_add_pedigree_info.py で使用される .json ファイル（血統キャッシュ、処理済みURLリスト）について、定期的に更新・整理を行う機能を追加予定です。

- 必要に応じてWeb UI化なども検討していく方針です。
---

## 📮 作者について
本プロジェクトは、ChatGPTの支援を大いに活用して構築されています。
細部のアルゴリズムやコード実装に関しては、現時点で完全に理解が追いついていない部分もありますが、
今後の継続的な学習を通じて、これらの技術的な理解を深めていく予定です。
このプロジェクトは、データ分析・機械学習の学習を目的とした自主制作プロジェクトです。
実務での利用経験はありませんが、スクレイピングからモデル構築・予測までを一貫して構築する力を証明するためのものです。

何かご意見・ご質問がありましたら、お気軽にご連絡ください。

