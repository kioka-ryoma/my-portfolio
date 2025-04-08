import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import re
import os
import json
from pathlib import Path

class HorseDataProcessor:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.data_cache_file = self.checkpoint_dir / "horse_data_cache.json"
        self.processed_urls_file = self.checkpoint_dir / "processed_urls.json"
        self.batch_size = 1000
        self.horse_cache = self.load_cache()
        self.processed_urls = self.load_processed_urls()
        
    def load_cache(self):
        """キャッシュされた馬データを読み込む"""
        if self.data_cache_file.exists():
            with open(self.data_cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def load_processed_urls(self):
        """処理済みURLを読み込む"""
        if self.processed_urls_file.exists():
            with open(self.processed_urls_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_cache(self):
        """馬データのキャッシュを保存"""
        with open(self.data_cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.horse_cache, f, ensure_ascii=False, indent=2)
            
    def save_processed_urls(self):
        """処理済みURLを保存"""
        with open(self.processed_urls_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_urls, f, ensure_ascii=False, indent=2)

    def extract_race_record_from_text(self, text):
        pattern = r"(\d+)戦(\d+)勝"
        match = re.search(pattern, text)
        if match:
            total_races = int(match.group(1))
            wins = int(match.group(2))
            win_rate = (wins / total_races) * 100 if total_races > 0 else 0
            return total_races, wins, round(win_rate, 2)
        return None, None, None

    def get_horse_data_via_requests(self, horse_url, session):
        # URLが処理済みの場合はキャッシュから取得
        if horse_url in self.horse_cache:
            return self.horse_cache[horse_url]
            
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            res = session.get(horse_url, headers=headers)
            res.encoding = res.apparent_encoding
            if res.status_code != 200:
                return None
            
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # 通算成績抽出
            race_record = None
            prof_table = soup.select_one('.db_prof_table')
            if prof_table:
                rows = prof_table.select('tr')
                for row in rows:
                    if "通算成績" in row.text:
                        td = row.find('td')
                        if td:
                            race_record = td.text.strip()
                        break
            
            total_races, wins, win_rate = self.extract_race_record_from_text(race_record) if race_record else (None, None, None)
            
            # 血統情報抽出
            pedigree = {"父":"", "父父":"", "父母":"", "母":"", "母父":"", "母母":""}
            blood_table = soup.select_one('.blood_table')
            if blood_table:
                a_tags = blood_table.select('a')
                if len(a_tags) > 0: pedigree["父"] = a_tags[0].text.strip()
                if len(a_tags) > 1: pedigree["父父"] = a_tags[1].text.strip()
                if len(a_tags) > 2: pedigree["父母"] = a_tags[2].text.strip()
                if len(a_tags) > 3: pedigree["母"] = a_tags[3].text.strip()
                if len(a_tags) > 4: pedigree["母父"] = a_tags[4].text.strip()
                if len(a_tags) > 5: pedigree["母母"] = a_tags[5].text.strip()
            
            horse_data = {
                "総レース数": total_races,
                "勝利数": wins,
                "勝率": win_rate,
                **pedigree
            }
            
            # キャッシュに保存
            self.horse_cache[horse_url] = horse_data
            return horse_data
            
        except Exception as e:
            print(f"データ取得エラー: {str(e)}")
            return None

    def select_input_files(self):
        root = tk.Tk()
        root.withdraw()
        file_paths = filedialog.askopenfilenames(
            title="レース結果のCSVファイルを選択してください",
            filetypes=[("CSV files", "*.csv")]
        )
        return file_paths

    def save_batch_results(self, merged_df, batch_num):
        """バッチ処理結果の保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'horse_data_batch_{batch_num}_{timestamp}.csv'
        merged_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"バッチ {batch_num} の結果を {output_filename} に保存しました")

    def process_horse_data(self):
        # ファイル選択
        file_paths = self.select_input_files()
        if not file_paths:
            print("ファイルが選択されませんでした。")
            return None
        
        # CSVファイルの読み込みと結合
        dfs = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                dfs.append(df)
                print(f"ファイル読み込み成功: {file_path}")
            except Exception as e:
                print(f"ファイル読み込みエラー {file_path}: {str(e)}")
        
        if not dfs:
            print("有効なデータが読み込めませんでした。")
            return None
        
        race_results_df = pd.concat(dfs, ignore_index=True)
        
        session = requests.Session()
        
        # 重複のない馬URL-馬名のペアを取得
        unique_horses = race_results_df[['馬名', '馬URL']].drop_duplicates().values.tolist()
        horse_data_list = []
        error_horses = []
        
        print(f"\n全{len(unique_horses)}頭の馬データを取得します...")
        
        # バッチ処理用の変数
        current_batch = []
        batch_num = 1
        
        for i, (horse_name, horse_url) in enumerate(unique_horses, 1):
            print(f"処理中 ({i}/{len(unique_horses)}): {horse_name}")
            
            # 既に処理済みのURLの場合はスキップ
            if horse_url in self.processed_urls:
                print(f"スキップ: {horse_name} (既に処理済み)")
                if horse_url in self.horse_cache:
                    horse_data = self.horse_cache[horse_url]
                    horse_data["馬名"] = horse_name
                    current_batch.append(horse_data)
                continue
            
            horse_data = self.get_horse_data_via_requests(horse_url, session)
            
            if horse_data:
                horse_data["馬名"] = horse_name
                current_batch.append(horse_data)
                self.processed_urls.append(horse_url)
                print(f"成功: {horse_name}")
            else:
                error_horses.append(horse_name)
                print(f"失敗: {horse_name} (データ取得エラー)")
            
            # バッチサイズに達したら中間保存
            if len(current_batch) >= self.batch_size:
                batch_df = pd.DataFrame(current_batch)
                self.save_batch_results(batch_df, batch_num)
                self.save_cache()
                self.save_processed_urls()
                horse_data_list.extend(current_batch)
                current_batch = []
                batch_num += 1
            
            time.sleep(1.0)
        
        # 残りのデータを保存
        if current_batch:
            batch_df = pd.DataFrame(current_batch)
            self.save_batch_results(batch_df, batch_num)
            horse_data_list.extend(current_batch)
        
        if not horse_data_list:
            print("血統データを取得できませんでした。")
            return None
        
        # 最終的なデータフレームの作成と保存
        pedigree_df = pd.DataFrame(horse_data_list)
        merged_df = pd.merge(race_results_df, pedigree_df, on='馬名', how='left')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'complete_horse_data_{timestamp}.csv'
        merged_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        
        # エラーログの保存
        if error_horses:
            with open(f'error_horses_{timestamp}.txt', 'w', encoding='utf-8') as f:
                for horse in error_horses:
                    f.write(f"{horse}\n")
        
        # 最終的なキャッシュの保存
        self.save_cache()
        self.save_processed_urls()
        
        print("\n処理完了:")
        print(f"成功: {len(horse_data_list)} 頭")
        print(f"失敗: {len(error_horses)} 頭")
        print(f"最終結果は {output_filename} に保存されました")
        
        return merged_df

if __name__ == "__main__":
    processor = HorseDataProcessor()
    processor.process_horse_data()
