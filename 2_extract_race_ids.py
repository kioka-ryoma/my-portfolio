import requests
from bs4 import BeautifulSoup
import re
import time
import random

def get_race_urls_from_page(base_url, page):
    # ページ番号をURLに追加
    url = f"{base_url}&page={page}" if page > 1 else base_url
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # リクエストの送信（念のため少し待機）
        time.sleep(random.uniform(1, 2))
        response = requests.get(url, headers=headers)
        response.encoding = response.apparent_encoding

        if response.status_code != 200:
            print(f"エラー: ステータスコード {response.status_code} (ページ {page})")
            return []

        # BeautifulSoupで解析
        soup = BeautifulSoup(response.text, 'html.parser')
        race_rows = soup.find_all('tr')
        race_urls = []

        for row in race_rows:
            race_name_td = row.find('td', class_='txt_l', attrs={'nowrap': 'nowrap'})
            if race_name_td:
                race_link = race_name_td.find('a')
                if race_link and 'href' in race_link.attrs:
                    url = race_link['href']
                    if re.match(r'^/race/\d{12}/$', url):
                        # URLから数字のみを抽出
                        race_id = re.search(r'\d{12}', url).group()
                        race_urls.append(race_id)

        return race_urls

    except requests.RequestException as e:
        print(f"エラーが発生しました (ページ {page}): {e}")
        return []

def scrape_all_pages(base_url, max_pages):
    all_race_ids = []
    
    for page in range(1, max_pages + 1):
        print(f"ページ {page} をスクレイピング中...")
        page_urls = get_race_urls_from_page(base_url, page)
        
        if not page_urls:
            print(f"ページ {page} でデータが取得できませんでした。終了します。")
            break
            
        all_race_ids.extend(page_urls)
        
    return all_race_ids

def save_to_file(race_ids, filename="race_ids_2023.txt"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # 20個ごとに改行を入れて整形
            for i in range(0, len(race_ids), 20):
                chunk = race_ids[i:i + 20]
                formatted_chunk = ', '.join(f'"{id}"' for id in chunk)
                f.write(formatted_chunk + ',\n' if i + 20 < len(race_ids) else formatted_chunk)
        print(f"データを {filename} に保存しました。")
    except IOError as e:
        print(f"ファイル保存中にエラーが発生しました: {e}")

def main():
    base_url = "https://db.netkeiba.com/?pid=race_list&word=&start_year=2025&start_mon=1&end_year=2025&end_mon=12&jyo%5B%5D=01&jyo%5B%5D=02&jyo%5B%5D=03&jyo%5B%5D=04&jyo%5B%5D=05&jyo%5B%5D=06&jyo%5B%5D=07&jyo%5B%5D=08&jyo%5B%5D=09&jyo%5B%5D=10&kyori_min=&kyori_max=&sort=date&list=100"
    
    # スクレイピングするページ数を指定
    max_pages = int(input("スクレイピングするページ数を入力してください: "))
    
    # スクレイピングの実行
    race_ids = scrape_all_pages(base_url, max_pages)
    
    if race_ids:
        print(f"\n合計: {len(race_ids)}件のレースIDを取得しました。")
        
        # ファイルに保存
        save_to_file(race_ids)
    else:
        print("データを取得できませんでした。")

if __name__ == "__main__":
    main()
