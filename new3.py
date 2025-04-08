import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
import re
import pickle
from datetime import datetime

# ==============================
# ヘルパー関数
# ==============================
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
        return {key: np.nan for key in [
            '年', '月', '日', '曜日', '四半期', '週番号',
            '月_sin', '月_cos', '日_sin', '日_cos', '経過日数'
        ]}

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
    """外れ値処理用の関数"""
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    return df

def find_best_threshold_from_array(y_true, y_proba):
    """
    与えられた予測確率と正解ラベルから、F1スコアを最大化する最適閾値を探索する関数
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_threshold, best_f1

def calculate_ensemble_weight(y_true, y_proba):
    """
    F1, ROC-AUC, Recall の平均値を重みとして採用
    """
    y_pred = (y_proba >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    recall = recall_score(y_true, y_pred)
    return (f1 + auc + recall) / 3

def save_model(model_results, filename='race_model_optimized.pkl'):
    """モデルの保存"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model_results, f)
        print(f"モデルを {filename} に保存しました。")
    except Exception as e:
        print(f"モデルの保存中にエラーが発生しました: {str(e)}")

# ==============================
# 前処理
# ==============================
def preprocess_data(df):
    """データの前処理関数（修正版）"""
    print(f"元のデータ行数: {len(df)}")
    
    # ターゲット変数: 上位3着以内を1、それ以外を0
    def parse_rank(rank):
        if pd.isna(rank) or rank == '---':
            return np.nan
        try:
            rank = int(rank)
            return 1 if rank <= 3 else 0
        except ValueError:
            return np.nan

    # 着順_original を数値として保持
    if '着順' in df.columns:
        df['着順_original'] = pd.to_numeric(df['着順'], errors='coerce')
    
    df['ターゲット'] = df['着順'].apply(parse_rank)
    
    # 数値列の処理
    numeric_cols = ['単勝', '総レース数', '斤量']
    for col in numeric_cols:
        df[col] = df[col].apply(clean_numeric_data)
    
    # 相対単勝オッズの計算
    print("\n相対単勝オッズの計算開始...")
    df['相対単勝オッズ'] = df.groupby('URL')['単勝'].transform(
        lambda x: x / x.mean() if x.mean() != 0 else 1.0
    )
    
    df['勝率'] = df['勝率'].apply(lambda x: clean_numeric_data(x)/100 if x != '---' and pd.notna(x) else np.nan)

    # 馬体重処理
    print("\n馬体重データの処理開始...")
    weight_data = df['馬体重'].apply(extract_weight_and_change)
    df['馬体重'] = weight_data.apply(lambda x: x[0])
    df['馬体重変化'] = weight_data.apply(lambda x: x[1])
    
    # 距離抽出
    def extract_distance(distance_str):
        if pd.isna(distance_str) or distance_str == '---':
            return np.nan
        match = re.search(r'(\d+)m', str(distance_str))
        return int(match.group(1)) if match else np.nan

    df['距離_数値'] = df['距離'].apply(extract_distance)

    # 年齢・性別抽出
    def extract_age(sex_age):
        if pd.isna(sex_age) or sex_age == '---':
            return np.nan
        match = re.search(r'\d+', str(sex_age))
        return int(match.group()) if match else np.nan

    def extract_sex(sex_age):
        if pd.isna(sex_age) or sex_age == '---':
            return np.nan
        match = re.match(r'[^\d]+', str(sex_age))
        return match.group() if match else np.nan

    df['年齢'] = df['性齢'].apply(extract_age)
    df['性別'] = df['性齢'].apply(extract_sex)
    
    # 外れ値処理
    numerical_columns = ['距離_数値', '斤量', '馬体重', '馬体重変化', '年齢', '相対単勝オッズ']
    df = handle_outliers(df, numerical_columns)

    # カテゴリカル変数のエンコード
    categorical_columns = [
        '騎手番号','枠番','天候', '馬場状態', '性別',
        '馬名', '出馬数','父', '父父', '父母', '母', '母父', '母母'
    ]
    encoded_columns = []
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('missing').replace('---', 'missing')
            df[col + '_encoded'] = df[col].astype('category').cat.codes
            encoded_columns.append(col + '_encoded')

    # 人気を数値型に変換
    df['人気'] = pd.to_numeric(df['人気'], errors='coerce')
    df['人気_カテゴリ'] = pd.Categorical(df['人気']).codes

    # 数値カラムのリスト
    num_cols_for_scaling = [
        '距離_数値', '斤量', '馬体重', '馬体重変化',
        '年齢', '単勝', '相対単勝オッズ'
    ]

    # 欠損値補完（中央値）
    for col in num_cols_for_scaling:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # 開催日の特徴量追加
    date_features = df['開催日'].apply(extract_date_features).apply(pd.Series)
    df = pd.concat([df, date_features], axis=1)
    
    new_date_numericals = [
        '年', '月', '日', '曜日',
        '四半期', '週番号', '月_sin', '月_cos', '日_sin', '日_cos', '経過日数'
    ]
    num_cols_for_scaling.extend(new_date_numericals)

    # 標準化（人気は除外）
    scaler = StandardScaler()
    df[num_cols_for_scaling] = scaler.fit_transform(df[num_cols_for_scaling])
    
    # 不要カラムの削除（ただし、着順_originalは保持）
    drop_columns = [
        'レース名', 'URL', '馬URL', '距離', '性齢', '騎手', '騎手URL',
        '着順', '勝利数', '勝率', '開催日'
    ] + categorical_columns
    df = df.drop(columns=drop_columns, errors='ignore')
    
    # 最終欠損値除去
    df = df.dropna()
    print(f"\n前処理後のデータ行数: {len(df)}")
    
    return df, encoded_columns, num_cols_for_scaling

# ==============================
# モデル構築用関数（LightGBM のみ）
# ==============================
def build_single_model(class_weight):
    """
    LightGBM のモデルを返す。
    GPU利用、device='gpu' を指定。
    """
    return LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        random_state=42,
        verbose=-1,
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=class_weight,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        n_jobs=1
    )

# ==============================
# 反復学習・更新および重み付きアンサンブル（LightGBM のみ）
# ==============================
def train_model_iterative(file_path, save_path='race_model_iterative.pkl', max_rounds=3):
    """
    反復学習で、以下の定義に基づく誤分類サンプルのみを抽出し再学習する手法：
      - 各ラウンドで、最適閾値 T を算出し、予測が T 以上なのに実際の順位が5位以上、または
        予測が T 未満なのに実際の順位が3位以下の場合を誤分類とする。
    また、再学習データには元の訓練データからランダムに10%を追加し、
    ラウンド1で LGBM のハイパーパラメータチューニングをGPU利用で実施します。
    最終的に、各ラウンドの評価指標（F1, AUC, Recall の平均）を重みとしてアンサンブルを行います。
    """
    print("データ読み込み開始...")
    data = pd.read_csv(file_path)
    processed_data, cat_features, num_features = preprocess_data(data)
    
    # 全特徴量（着順_original を残す）とターゲット
    X_all = processed_data.drop(columns=['ターゲット'])
    y_all = processed_data['ターゲット']
    
    # train/test split（テストセットは固定）
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    # 訓練データから、誤分類抽出用に着順_original を分離
    X_train_features = X_train.drop(columns=['着順_original'])
    train_ranks = X_train['着順_original']
    
    ensemble_models = []  # 各モデルのタプル (name, model, weight)
    round_details = []
    
    # current_X, current_y, current_ranks は各ラウンドの再学習データ
    current_X = X_train_features.copy()
    current_y = y_train.copy()
    current_ranks = train_ranks.copy()
    
    for r in range(1, max_rounds + 1):
        print(f"\n================ Round {r} =================")
        num_pos = current_y.sum()
        num_neg = len(current_y) - num_pos
        class_weight = (num_neg / num_pos) if num_pos != 0 else 1.0
        
        # ラウンド1 ではハイパーパラメータチューニングを実施
        if r == 1:
            param_dist = {
                'n_estimators': [500, 1000, 1500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5],
                'min_child_samples': [20, 50, 100],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0]
            }
            base_model = LGBMClassifier(
                scale_pos_weight=class_weight,
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                random_state=42,
                n_jobs=1
            )
            search = RandomizedSearchCV(
                base_model,
                param_dist,
                n_iter=10,
                scoring='f1',
                cv=3,
                random_state=42,
                n_jobs=1
            )
            search.fit(current_X, current_y)
            best_model = search.best_estimator_
            print(f"    -> LGBM 最適パラメータ: {search.best_params_}")
            model = best_model
        else:
            model = build_single_model(class_weight)
        
        # ここでは早期終了パラメータを使用せずに通常 fit を実施
        model.fit(current_X, current_y)
            
        model_name = f"lgb_round{r}"
        ensemble_models.append((model_name, model, None))  # 後で重みを設定
        
        # ラウンド内での予測確率の平均（今回は1モデルのみなのでそのまま）
        preds_proba = model.predict_proba(current_X)[:, 1]
        
        # 改善案1：各ラウンドで最適閾値を探索
        best_threshold, best_f1_round = find_best_threshold_from_array(current_y, preds_proba)
        print(f"Round {r} 最適閾値: {best_threshold:.3f} （F1: {best_f1_round:.4f}）")
        
        # 誤分類の抽出（最適閾値を用いる）
        misclassified_idx = []
        for i, (p, rank) in enumerate(zip(preds_proba, current_ranks)):
            if p >= best_threshold and rank >= 5:
                misclassified_idx.append(i)
            elif p < best_threshold and rank <= 3:
                misclassified_idx.append(i)
        misclassified_idx = np.array(misclassified_idx)
        
        print(f"Round {r}: 全サンプル数 {len(current_y)} 中、誤分類サンプル数: {len(misclassified_idx)}")
        y_pred_round = (preds_proba >= best_threshold).astype(int)
        print(f"Round {r} - 分類レポート:")
        print(classification_report(current_y, y_pred_round))
        print(f"Round {r} - 混同行列:")
        print(confusion_matrix(current_y, y_pred_round))
        if len(misclassified_idx) > 0:
            misclassified_ranks = current_ranks.iloc[misclassified_idx]
            print("誤分類サンプルの実際の順位分布:")
            print(misclassified_ranks.value_counts())
        else:
            print("誤分類サンプルは見つかりませんでした。以降の更新は行いません。")
            break
        
        # 改善案2：F1, AUC, Recall の平均を重みとして採用
        round_weight = calculate_ensemble_weight(current_y, preds_proba)
        print(f"Round {r} の重み（F1, AUC, Recall 平均）: {round_weight:.4f}")
        
        # ensemble_models 内、該当ラウンドのモデルに重みを設定
        for i in range(len(ensemble_models)):
            name, model_tmp, _ = ensemble_models[i]
            if f"round{r}" in name:
                ensemble_models[i] = (name, model_tmp, round_weight)
        
        # 改善案3：次ラウンド用に、誤分類サンプルに加え元データの10%を追加
        misclassified_X = current_X.iloc[misclassified_idx]
        misclassified_y = current_y.iloc[misclassified_idx]
        misclassified_ranks = current_ranks.iloc[misclassified_idx]
        
        remaining_idx = X_train_features.index.difference(misclassified_X.index)
        additional_samples = X_train_features.loc[remaining_idx].sample(frac=0.2, random_state=42)
        
        current_X = pd.concat([misclassified_X, additional_samples])
        current_y = pd.concat([misclassified_y, y_train.loc[additional_samples.index]])
        current_ranks = pd.concat([misclassified_ranks, train_ranks.loc[additional_samples.index]])
        
        current_X = current_X.reset_index(drop=True)
        current_y = current_y.reset_index(drop=True)
        current_ranks = current_ranks.reset_index(drop=True)
        
        round_details.append({
            'round': r,
            'num_samples': len(current_y),
            'ensemble_weight': round_weight,
            'misclassified_count': len(misclassified_idx)
        })
        
        if len(current_y) == 0:
            print("更新対象サンプルがなくなったため、反復学習を終了します。")
            break
        
        print("-------------------------------------------------------")
    
    # 最終テストデータに対する重み付きアンサンブル予測
    print("\n=== 最終テストデータ評価 ===")
    X_test_features = X_test.drop(columns=['着順_original'])
    total_weight = sum([w if w is not None else 1 for (_, _, w) in ensemble_models])
    preds_weighted = np.zeros(len(X_test_features))
    for (name, model, w) in ensemble_models:
        weight = w if w is not None else 1
        preds_weighted += model.predict_proba(X_test_features)[:, 1] * weight
    preds_weighted /= total_weight
    final_preds = (preds_weighted >= 0.5).astype(int)
    
    print("最終テストデータ - 分類レポート:")
    print(classification_report(y_test, final_preds))
    print("最終テストデータ - 混同行列:")
    print(confusion_matrix(y_test, final_preds))
    auc_score = roc_auc_score(y_test, preds_weighted)
    print(f"最終 ROC-AUC Score: {auc_score:.4f}")
    
    model_results = {
        'ensemble_models': ensemble_models,
        'final_predictions': final_preds,
        'final_probability': preds_weighted,
        'test_y': y_test,
        'round_details': round_details
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_results, f)
    print(f"\n最終モデルを {save_path} に保存しました。")
    
    return model_results

# ==============================
# スクリプト実行時のエントリポイント
# ==============================
if __name__ == "__main__":
    file_path = "C:/python/python_n/6years.csv"  # CSVファイルへのパス
    save_path = "race_model_iterative3.pkl"
    try:
        results = train_model_iterative(file_path, save_path, max_rounds=3)
        print("\n反復学習によるモデルの学習が完了し、保存されました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
