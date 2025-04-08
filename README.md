# my-portfolio

# 🏈 競馬レース予測モデル構築プロジェクト

このリポジトリでは、**競馬レースの結果予測モデルを構築・適用するための一連のスクリプト群**を提供しています。  
スクレイピングから前処理・モデル学習・推論まで、**実務でも活用できるフロー**をPythonで実装しています。

---

## 📁 ファイル構成

| ファイル名 | 説明 |
|------------|------|
| `1_scrape_race_data.py` | モデル学習用の過去レースデータを取得（スクレイピング） |
| `2_extract_race_ids.py` | 取得対象レースIDのリストを自動で生成 |
| `3_add_pedigree_info.py` | レースデータに血統情報を付加 |
| `4_train_race_model.py` | LightGBMを用いたモデルの訓練＋アンサンブル |
| `5_scrape_prediction_race.py` | 予測対象レースの最新出馬情報を取得 |
| `6_predict_race_results.py` | 学習済みモデルと最新データを用いた推論の実行 |

---

## 🚀 プロジェクトの流れ

### 1. データ取得（過去レース）
- `2_extract_race_ids.py`でレースIDを取得  
- `1_scrape_race_data.py`で対象レースの詳細データをスクレイピング

### 2. 血統情報の付与
- `3_add_pedigree_info.py`で、出走馬ごとの血統と通算成績を付加しCSV化

### 3. モデルの構築
- `4_train_race_model.py`でLightGBMによる分類モデルを構築（3着以内を1と分類）
- 誤分類データに基づく反復学習、各ラウンドごとに重み付けしアンサンブル学習を実施

### 4. 予測対象レースのデータ取得
- `5_scrape_prediction_race.py`で、直近レースの出走情報＋血統データを取得・整形

### 5. モデルによる推論
- `6_predict_race_results.py`で、出力されたCSVに対して推論を実行
- 上位入着確率とZスコアを表示
- 騎手別の分析やCSVへの出力も自動化

---

## 📊 使用技術・ライブラリ

- **スクレイピング**：`requests`, `BeautifulSoup`, `selenium`
- **データ前処理**：`pandas`, `numpy`, `re`, `datetime`
- **モデル学習**：`LightGBM`, `scikit-learn`
- **モデル保存**：`pickle`

---



## 📆 予測結果の例

```
馬名: トウカイテイオー, 騎手: 福永祇一, 予測確率: 87.2%, 標準化スコア: 2.15
馬名: ナリタブライアン, 騎手: 武豪,      予測確率: 83.4%, 標準化スコア: 1.87
馬名: スペシャルウィーク, 騎手: 横山具弘, 予測確率: 78.5%, 標準化スコア: 1.53
```

---

## 💬 使用方法

1. 学習データ作成
   ```
   python 2_extract_race_ids.py
   python 1_scrape_race_data.py
   python 3_add_pedigree_info.py
   ```
2. モデル訓練
   ```
   python 4_train_race_model.py
   ```
3. 予測用レース情報取得
   ```
   python 5_scrape_prediction_race.py
   ```
4. 推論実行
   ```
   python 6_predict_race_results.py
   ```

---

