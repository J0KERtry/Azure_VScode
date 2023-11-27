from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import schedule
import datetime

### 定数の定義 ###
INSTAGRAM_URL = 'https://www.instagram.com/accounts/login/' # インスタにログインする際のURL
MY_URL = 'https://www.instagram.com/direct/t/17844455750312990/' # 自分に送信するテストURL
# RINA_URL = 'https://www.instagram.com/direct/t/108603177202885/'  # 受信者とのチャットページのURL
USERNAME = 'yudai_sub_0t9' # 送信者のユーザーネーム
PASSWORD = 'yudai_0t9' # 送信者のパスワード
USER_AGENT1 = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
USER_AGENT2 = 'Mozilla/5.0 (X11; CrOS x86_64 10066.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
message_index = 0
counter = 0
    
# main関数
def main():
    global counter
    global message_index
    url = MY_URL # 送信する人選択

    # もし深夜0時から朝9時の間なら送信
    now = datetime.datetime.fromtimestamp(time.time())
    if now.hour >= 11 and now.hour < 20:
    ### ブラウザの設定 ###
        chrome_options = Options()
        chrome_options.binary_location = r'C:\Program Files\Google\Chrome\Application\chrome.exe'  # Chromeのパス
        driver_path = r'C:\Users\Yudai\OneDrive\デスクトップ\message\chromedriver-win64\chromedriver.exe'  # ChromeDriverのパス

        env = USER_AGENT1 if counter % 2 == 1 else USER_AGENT2
        chrome_options.add_argument('--user-agent=' + env)
        browser = webdriver.Chrome(options=chrome_options)
        counter += 1


    ### ログイン ###
        browser.get(INSTAGRAM_URL)     
        # ユーザーネームとパスワードフィールドが利用可能になるまで待つ
        WebDriverWait(browser, 6).until(EC.presence_of_element_located((By.NAME, 'username')))
        WebDriverWait(browser, 6).until(EC.presence_of_element_located((By.NAME, 'password')))
        username_field = browser.find_element(By.NAME, 'username')
        password_field = browser.find_element(By.NAME, 'password')
        username_field.send_keys(USERNAME)
        time.sleep(2)
        password_field.send_keys(PASSWORD + Keys.ENTER)
        time.sleep(3)
        
    ### メッセージ一覧開く ###
        button = browser.find_element(By.CSS_SELECTOR, 'svg.x1lliihq.x1n2onr6.x5n08af')
        button.click()
        
    ### 個別のメッセージ画面開く ###
        browser.execute_script('window.location.href = "{}"'.format(url))

    ### メッセージ取得 ###
        with open('messages.txt', 'r', encoding='utf-8') as f:
            MESSAGES = f.read().splitlines()
        message = MESSAGES[message_index]
        message_index += 1

    ### メッセージ入力 ###
        text_box = browser.find_element(By.XPATH, '//div[@contenteditable="true"]')
        text_box.send_keys(message + Keys.ENTER)
        time.sleep(5)

        browser.quit()



### スケジュールの設定 ###
try:
    # 自動化スクリプトと判断されないための実行
    # 送信したい時間（分）を設定
    schedule.every(0.1).minutes.do(main)
except TypeError:
    pass

### スケジュールの実行 ###
try:
    while True:
        schedule.run_pending()
        # ローカルPCを休ませる（分）
        time.sleep(0.1)
except UnboundLocalError:
    pass
