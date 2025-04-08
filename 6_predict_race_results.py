import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import re
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog

class CustomEnsemblePredictor:
    """
    カスタム予測アンサンブルクラス
    VotingClassifierを使用せずに、直接予測確率の平均を計算
    """
    def __init__(self, models):
        self.models = models  # [(name, model), ...]
    
    def predict_proba(self, X):
        """
        全モデルの予測確率の平均を計算
        """
        all_probas = []
        for name, model in self.models:
            proba = model.predict_proba(X)
            all_probas.append(proba)
        avg_proba = np.mean(all_probas, axis=0)
        return avg_proba
    
    def predict(self, X, threshold=0.5):
        """
        指定された閾値で予測を行う
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= threshold).astype(int)

def load_model(model_path='race_model.pkl'):
    """保存されたモデルを読み込む関数"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model_version = getattr(model_data, 'version', 'unknown')
            print(f"モデルバージョン: {model_version}")
            return model_data
    except Exception as e:
        print(f"モデルの読み込み中にエラーが発生しました: {str(e)}")
        return None

def extract_date_features(date_str):
    """開催日から特徴量を抽出する関数"""
    try:
        date_obj = datetime.strptime(str(date_str), '%Y%m%d')
        features = {
            '年': date_obj.year,
            '月': date_obj.month,
            '日': date_obj.day,
            '曜日': date_obj.weekday(),
            '四半期': (date_obj.month-1) // 3 + 1,
            '週番号': date_obj.isocalendar()[1],
            '月_sin': np.sin(2 * np.pi * date_obj.month / 12),
            '月_cos': np.cos(2 * np.pi * date_obj.month / 12),
            '日_sin': np.sin(2 * np.pi * date_obj.day / 31),
            '日_cos': np.cos(2 * np.pi * date_obj.day / 31),
            '経過日数': (date_obj - datetime(2019, 1, 1)).days
        }
        return features
    except (ValueError, TypeError):
        return {key: np.nan for key in ['年', '月', '日', '曜日', '四半期', '週番号', 
                                         '月_sin', '月_cos', '日_sin', '日_cos', '経過日数']}

def clean_numeric_data(value):
    """数値データのクリーニング"""
    if pd.isna(value) or value == '---':
        return np.nan
    try:
        return float(str(value).replace(',', ''))
    except (ValueError, TypeError):
        return np.nan

def extract_weight_and_change(weight_str):
    """馬体重と変化量を抽出"""
    if pd.isna(weight_str) or weight_str == '---':
        return np.nan, np.nan
    try:
        weight = int(weight_str.split('(')[0])
        change_str = weight_str.split('(')[1].rstrip(')')
        if change_str.startswith('+'):
            change = int(change_str[1:])
        elif change_str.startswith('-'):
            change = -int(change_str[1:])
        else:
            change = int(change_str)
        return weight, change
    except (ValueError, IndexError):
        return np.nan, np.nan

def handle_outliers(df, columns):
    """外れ値処理"""
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    return df

def preprocess_prediction_data(df):
    """予測用データの前処理"""
    try:
        required_columns = ['開催日', '単勝', '馬体重', '性齢', '距離', '騎手']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"必要なカラムが欠落: {', '.join(missing_columns)}")

        # 開催日特徴量
        date_features = df['開催日'].apply(extract_date_features).apply(pd.Series)
        df = pd.concat([df, date_features], axis=1)

        # 数値データ処理
        numeric_cols = ['単勝', '総レース数', '斤量']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric_data)

        # 相対単勝オッズ計算（グループはレース名）
        df['相対単勝オッズ'] = df.groupby('レース名')['単勝'].transform(
            lambda x: x / x.mean() if x.mean() != 0 else 1.0
        )

        # 勝率の処理
        if '勝率' in df.columns:
            df['勝率'] = df['勝率'].apply(lambda x: clean_numeric_data(x)/100 if x != '---' and pd.notna(x) else np.nan)

        # 馬体重データの処理
        weight_data = df['馬体重'].apply(extract_weight_and_change)
        df['馬体重'] = weight_data.apply(lambda x: x[0])
        df['馬体重変化'] = weight_data.apply(lambda x: x[1])

        # 距離の処理
        df['距離_数値'] = df['距離'].apply(
            lambda x: int(re.search(r'(\d+)m', str(x)).group(1)) if pd.notna(x) and x != '---' else np.nan
        )

        # 性齢の処理
        df['年齢'] = df['性齢'].apply(
            lambda x: int(re.search(r'\d+', str(x)).group()) if pd.notna(x) and x != '---' else np.nan
        )
        df['性別'] = df['性齢'].apply(
            lambda x: re.match(r'[^\d]+', str(x)).group() if pd.notna(x) and x != '---' else np.nan
        )

        # カテゴリカル変数のエンコード
        categorical_columns = [
            '騎手番号','枠番','天候', '馬場状態', '性別',
            '馬名', '出馬数','父', '父父', '父母', '母', '母父', '母母'
        ]
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('missing')
                df[col] = df[col].replace('---', 'missing')
                df[col + '_encoded'] = df[col].astype('category').cat.codes

        # 人気データの処理
        df['人気'] = pd.to_numeric(df['人気'], errors='coerce')
        df['人気_カテゴリ'] = pd.Categorical(df['人気']).codes

        # 数値カラムの処理
        num_cols_for_scaling = [
            '距離_数値', '斤量', '馬体重', '馬体重変化',
            '年齢', '単勝', '相対単勝オッズ',
            '年', '月', '日', '曜日', '四半期', '週番号',
            '月_sin', '月_cos', '日_sin', '日_cos', '経過日数'
        ]

        # 欠損値を中央値で補間
        for col in num_cols_for_scaling:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # 標準化
        scaler = StandardScaler()
        df[num_cols_for_scaling] = scaler.fit_transform(df[num_cols_for_scaling])

        # 不要なカラムの削除
        drop_columns = [
            'レース名', 'URL', '馬URL', '距離', '性齢', '騎手',
            '騎手URL', '勝利数', '勝率', '開催日'
        ] + [col for col in categorical_columns if col in df.columns]
        df = df.drop(columns=drop_columns, errors='ignore')

        # 特徴量の順序整理（存在する列のみ）
        expected_feature_order = ['斤量', '単勝', '人気', '馬体重', '総レース数', '相対単勝オッズ',
                                  '馬体重変化', '距離_数値', '年齢', '騎手番号_encoded', '枠番_encoded',
                                  '天候_encoded', '馬場状態_encoded', '性別_encoded', '馬名_encoded',
                                  '出馬数_encoded', '父_encoded', '父父_encoded', '父母_encoded',
                                  '母_encoded', '母父_encoded', '母母_encoded', '人気_カテゴリ',
                                  '年', '月', '日', '曜日', '四半期', '週番号', '月_sin', '月_cos',
                                  '日_sin', '日_cos', '経過日数']
        available_columns = [col for col in expected_feature_order if col in df.columns]
        df = df[available_columns]

        return df

    except Exception as e:
        print(f"前処理中にエラーが発生しました: {str(e)}")
        raise

def predict_races(data_path, model_path='race_model_iterative.pkl'):
    """レース予測を実行する関数"""
    try:
        # データの読み込みと前処理
        print(f"\n【{data_path}】の予測を開始します。")
        df = pd.read_csv(data_path)
        original_df = df.copy()

        model_results = load_model(model_path)
        if model_results is None:
            return None

        processed_df = preprocess_prediction_data(df)
        
        # モデルの構造に応じた予測処理
        if isinstance(model_results, dict):
            if 'ensemble_models' in model_results:
                # ensemble_models は (name, model, weight) のタプルリスト
                ensemble = CustomEnsemblePredictor([(name, model) for name, model, weight in model_results['ensemble_models']])
                raw_predictions = ensemble.predict_proba(processed_df)[:, 1]
            elif 'models' in model_results:
                ensemble = CustomEnsemblePredictor(model_results['models'])
                raw_predictions = ensemble.predict_proba(processed_df)[:, 1]
            elif 'ensemble' in model_results:
                raw_predictions = model_results['ensemble'].predict_proba(processed_df)[:, 1]
            elif 'model' in model_results:
                raw_predictions = model_results['model'].predict_proba(processed_df)[:, 1]
            else:
                # その他のケース
                first_key = list(model_results.keys())[0]
                print(f"利用可能なキー: {list(model_results.keys())}")
                if hasattr(model_results[first_key], 'predict_proba'):
                    raw_predictions = model_results[first_key].predict_proba(processed_df)[:, 1]
                else:
                    raise ValueError(f"予測可能なモデルが見つかりません。モデルの構造: {model_results.keys()}")
        else:
            # model_results自体がモデルの場合
            raw_predictions = model_results.predict_proba(processed_df)[:, 1]
            
        # 予測確率の調整
        predictions = (raw_predictions - raw_predictions.min()) / (raw_predictions.max() - raw_predictions.min())
        predictions = predictions * 0.8 + 0.1  # 0.1-0.9の範囲に調整
        
        z_scores = (predictions - np.mean(predictions)) / np.std(predictions)

        # 結果のデータフレーム作成
        results_df = pd.DataFrame({
            'レース名': original_df['レース名'],
            '馬名': original_df['馬名'],
            '騎手': original_df['騎手'],
            '予測確率': predictions * 100,
            '標準化スコア': z_scores.round(3),
            '予測': predictions > 0.5,  # デフォルトの閾値を使用
            '単勝オッズ': original_df['単勝'],
            '人気': original_df['人気']
        })

        return results_df.sort_values('標準化スコア', ascending=False)

    except Exception as e:
        print(f"予測エラー: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def save_predictions(results_df, output_path):
    """予測結果をCSVファイルとして保存する関数"""
    try:
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"race_predictions_{timestamp}.csv"
        full_path = os.path.join(output_path, filename)
        results_df.to_csv(full_path, index=False, encoding='utf-8')
        print(f"\n予測結果を保存しました: {full_path}")
    except Exception as e:
        print(f"予測結果の保存中にエラーが発生しました: {str(e)}")

def analyze_jockey_predictions(results_df):
    """騎手ごとの予測結果の分析"""
    try:
        jockey_analysis = results_df.groupby('騎手').agg({
            '予測確率': ['mean', 'count'],
            '標準化スコア': 'mean'
        }).round(3)
        
        print("\n騎手別分析:")
        print(jockey_analysis.sort_values(('標準化スコア', 'mean'), ascending=False))
        return jockey_analysis
    except Exception as e:
        print(f"騎手分析中にエラーが発生しました: {str(e)}")
        return None

if __name__ == "__main__":
    # tkinterのファイル選択ダイアログを利用して複数CSVファイルを選択する
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを表示しない

    file_paths = filedialog.askopenfilenames(
        title="予測したいCSVファイルを選択してください",
        filetypes=[("CSV files", "*.csv")]
    )

    if not file_paths:
        print("ファイルが選択されませんでした。処理を終了します。")
    else:
        # モデルのパスと予測結果保存先のパスを指定
        model_path = "race_model_iterative3.pkl"
        output_path = "predictions"

        # 選択された各CSVファイルについて処理を実行
        for file_path in file_paths:
            print("\n========================================")
            results = predict_races(file_path, model_path)
            
            if results is not None:
                print("\n予測結果:")
                print(results.to_string(index=False))
                
                # 推奨馬（Top 3）の表示
                print("\n推奨馬（Top 3）:")
                top_3 = results.head(3)
                for _, row in top_3.iterrows():
                    print(f"馬名: {row['馬名']}, 騎手: {row['騎手']}, "
                          f"標準化スコア: {row['標準化スコア']:.3f}, "
                          f"予測確率: {row['予測確率']:.2f}%")
                
                # 騎手分析の実行
                analyze_jockey_predictions(results)
                
                # 結果の保存
                save_predictions(results, output_path)
            else:
                print("予測を実行できませんでした。")
