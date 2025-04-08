from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from datetime import datetime
import time
import re

def parse_race_page(driver, url, race_date):
    """レースページから情報を取得する"""
    driver.get(url)
    time.sleep(1.5)  # ページ読み込みのための待機時間を追加
    
    # レース情報の取得
    race_info = {}
    
    try:
        race_name = driver.find_element(By.CLASS_NAME, "RaceName").text.strip()
        race_info['レース名'] = race_name

        race_data = driver.find_element(By.CLASS_NAME, "RaceData01")
        race_text = race_data.text.strip()
        race_info['距離'] = race_text.split('/')[1].strip() if '芝' in race_text else race_text.split('/')[1].strip()
        race_info['天候'] = race_text.split('天候:')[1].split('/')[0].strip() if '天候:' in race_text else "不明"
        race_info['馬場状態'] = race_text.split('馬場:')[1].strip() if '馬場:' in race_text else "不明"
        race_info['開催日'] = race_date

        # 出馬数を取得
        horse_rows = driver.find_elements(By.CSS_SELECTOR, "tr.HorseList")
        race_info['出馬数'] = len(horse_rows)
    except Exception as e:
        print(f"レース基本情報の取得に失敗: {str(e)}")
        return pd.DataFrame()

    results = []
    try:
        for row in horse_rows:
            try:
                frame = row.find_element(By.CSS_SELECTOR, "td[class^='Waku'] span").text.strip()
                
                horse_element = row.find_element(By.CSS_SELECTOR, "span.HorseName a")
                horse_name = horse_element.text.strip()
                horse_url = horse_element.get_attribute('href')
                
                sex_age = row.find_element(By.CLASS_NAME, "Barei").text.strip()
                weight = row.find_elements(By.TAG_NAME, "td")[5].text.strip()
                
                # 騎手情報の取得
                jockey_element = row.find_element(By.CLASS_NAME, "Jockey").find_element(By.TAG_NAME, "a")
                jockey_name = jockey_element.get_attribute('title').strip()
                jockey_url = jockey_element.get_attribute('href')
                # URLから/result/recentを削除して騎手番号を抽出
                jockey_url_base = jockey_url.replace('/result/recent', '')
                jockey_number = jockey_url_base.split('/')[-2]
                
                try:
                    odds = row.find_element(By.CSS_SELECTOR, "td.Popular span[id^='odds-']").text.strip()
                except:
                    odds = "---.-"
                    
                try:
                    popularity = row.find_element(By.CSS_SELECTOR, "td.Popular_Ninki span[id^='ninki-']").text.strip()
                except:
                    popularity = "**"
                
                body_weight = row.find_element(By.CLASS_NAME, "Weight").text.strip()
                
                results.append({
                    "レース名": race_info['レース名'],
                    "開催日": race_info['開催日'],
                    "距離": race_info['距離'],
                    "天候": race_info['天候'],
                    "馬場状態": race_info['馬場状態'],
                    "出馬数": race_info['出馬数'],  # 追加
                    "枠番": frame,
                    "馬名": horse_name,
                    "馬URL": horse_url,
                    "性齢": sex_age,
                    "斤量": weight,
                    "騎手": jockey_name,
                    "騎手URL": jockey_url_base,
                    "騎手番号": jockey_number,
                    "単勝": odds,
                    "人気": popularity,
                    "馬体重": body_weight,
                })
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
    except Exception as e:
        print(f"出走馬情報の取得に失敗: {str(e)}")
        return pd.DataFrame()
    
    return pd.DataFrame(results)
def extract_race_record(text):
    """通算成績を抽出する"""
    pattern = r"(\d+)戦(\d+)勝"
    match = re.search(pattern, text)
    if match:
        total_races = int(match.group(1))
        wins = int(match.group(2))
        win_rate = (wins / total_races) * 100 if total_races > 0 else 0
        return total_races, wins, round(win_rate, 2)
    return None, None, None

def get_horse_data(driver, horse_url):
    """個々の馬の詳細データを取得する"""
    try:
        driver.get(horse_url)
        profile_table = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "db_prof_table"))
        )
        
        race_record = None
        for row in profile_table.find_elements(By.TAG_NAME, "tr"):
            if "通算成績" in row.text:
                race_record = row.find_element(By.TAG_NAME, "td").text
                break
        
        total_races, wins, win_rate = extract_race_record(race_record) if race_record else (None, None, None)
        
        blood_table = driver.find_element(By.CLASS_NAME, "blood_table")
        blood_cells = blood_table.find_elements(By.TAG_NAME, "a")
        
        pedigree = {
            "父": blood_cells[0].text if len(blood_cells) > 0 else "",
            "父父": blood_cells[1].text if len(blood_cells) > 1 else "",
            "父母": blood_cells[2].text if len(blood_cells) > 2 else "",
            "母": blood_cells[3].text if len(blood_cells) > 3 else "",
            "母父": blood_cells[4].text if len(blood_cells) > 4 else "",
            "母母": blood_cells[5].text if len(blood_cells) > 5 else ""
        }
        
        return {
            "総レース数": total_races,
            "勝利数": wins,
            "勝率": win_rate,
            **pedigree
        }
        
    except Exception as e:
        print(f"データ取得エラー: {str(e)}")
        return None

def process_horse_data(driver, race_results_df):
    """全出走馬のデータを処理する"""
    unique_horses = race_results_df[['馬名', '馬URL']].drop_duplicates().values.tolist()
    horse_data_list = []
    error_horses = []
    
    print(f"\n全{len(unique_horses)}頭の馬データを取得します...")
    
    for i, (horse_name, horse_url) in enumerate(unique_horses, 1):
        print(f"処理中 ({i}/{len(unique_horses)}): {horse_name}")
        
        horse_data = get_horse_data(driver, horse_url)
        if horse_data:
            horse_data["馬名"] = horse_name
            horse_data_list.append(horse_data)
            print(f"成功: {horse_name}")
        else:
            error_horses.append(horse_name)
            print(f"失敗: {horse_name} (データ取得エラー)")
        
        time.sleep(1)
    
    pedigree_df = pd.DataFrame(horse_data_list)
    merged_df = pd.merge(race_results_df, pedigree_df, on='馬名', how='left')
    
    return merged_df, error_horses

def process_single_race(driver, url, race_date):
    """1つのレースURLに対する処理"""
    try:
        print(f"\nURLの処理を開始: {url}")
        
        # レース情報の取得
        print("レース情報の取得を開始します...")
        race_data = parse_race_page(driver, url, race_date)
        
        if not race_data.empty:
            print("\n出走馬情報の取得が完了しました。血統データの取得を開始します。")
            
            # 血統データの取得と結合
            final_df, error_horses = process_horse_data(driver, race_data)
            
            # レース名をファイル名に使用（ファイル名に使用できない文字を置換）
            safe_race_name = race_data['レース名'].iloc[0].replace('/', '').replace('\\', '').replace(':', '').replace('*', '').replace('?', '').replace('"', '').replace('<', '').replace('>', '').replace('|', '')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 結果の保存
            output_filename = f'{safe_race_name}_{timestamp}.csv'
            error_filename = f'{safe_race_name}_errors_{timestamp}.txt'
            
            final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            
            if error_horses:
                with open(error_filename, 'w', encoding='utf-8') as f:
                    for horse in error_horses:
                        f.write(f"{horse}\n")
            
            print("\n処理完了:")
            print(f"成功: {len(final_df['馬名'].unique())} 頭")
            print(f"失敗: {len(error_horses)} 頭")
            print(f"データが保存されました: {output_filename}")
            
            return True
        else:
            print("レースデータが見つかりませんでした。")
            return False
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return False

def main(urls, race_date):
    """メイン処理 - 複数URLの処理"""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-software-rasterizer')
    options.add_argument('--disable-webgl')
    options.add_argument('--disable-logging')
    options.add_argument('--log-level=3')
    options.add_argument('--silent')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=options)
    
    try:
        total_urls = len(urls)
        successful_urls = 0
        failed_urls = 0
        
        for i, url in enumerate(urls, 1):
            print(f"\n=== URL {i}/{total_urls} の処理を開始 ===")
            
            if process_single_race(driver, url, race_date):
                successful_urls += 1
            else:
                failed_urls += 1
        
        print("\n=== 全処理完了 ===")
        print(f"処理したURL数: {total_urls}")
        print(f"成功: {successful_urls}")
        print(f"失敗: {failed_urls}")
            
    finally:
        driver.quit()
if __name__ == "__main__":
    # スクレイピング対象のURL一覧
    target_urls = [
            

"https://race.netkeiba.com/race/shutuba.html?race_id=202509020312&rf=race_submenu",



        # 必要に応じてURLを追加
    ]
    
     # 開催日を指定 (YYYYMMDD形式)
    race_date = "20250329"  # 例: 2025年6月1日
    
    main(target_urls, race_date)


  
