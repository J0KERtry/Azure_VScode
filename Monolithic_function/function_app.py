# 月額料金[円] = { ( リソース使用量[GB/秒] - 400000[GB/秒] ) × 0.001792[円] } + { ( 実行回数[回] - 1000000[回] ) / 1000000 × 22.400[円] }

import azure.functions as func
import logging
import time
from collections import Counter

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="monolithic_functions")
def monolithic_functions(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # パラメータかjsonからデータ取得
    process = req.params.get('process') # 処理番号
    char = req.params.get('char')   # 置換文字
    string = req.params.get('string') # 長文
    if not process:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            process = req_body.get('process')
    if not char:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            char = req_body.get('char')
    if not string:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            string = req_body.get('string')

    # 文字列を結合して長文に変換
    n = 5
    long_string = ""
    for _ in range(n):
        long_string += " " + string

    # Activity関数群
    if process==100:
        return func.HttpResponse(f"Hello, {char}. This HTTP triggered function executed successfully.")
    
    #charを,設定した文字列に置換 (if文でさらに細分化し文字列を設定)
    elif process==0 and char and long_string:  
        if 'a' <= char[0] <= 'i':
            replacement = 'test1'
        elif 'j' <= char[0] <= 's':
            replacement = 'test2'
        else:
            replacement = 'test3'
        long_string = long_string.replace(char, replacement)

    #charを削除        
    elif process==1 and char and long_string:  
        long_string.replace(char, '')
    
    #charの出現回数
    elif process==2 and char and long_string:  
        char_count = long_string.count(char)
        return func.HttpResponse(f"The character '{char}' appears {char_count} times.")
    
    #回数ソート
    elif process==3 and long_string: 
        words = long_string.split()
        word_counts = Counter(words)
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        result = '\n'.join([f"{word}: {count}" for word, count in sorted_word_counts])
        return func.HttpResponse(result)
    
    # sleep
    elif process==4:    
        time.sleep(10)
    
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a string in the query stringing or in the request body for a personalized response.",
             status_code=200
        )