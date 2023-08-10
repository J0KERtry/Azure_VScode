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
    process = req.params.get('process')
    if process is None:
        try:
            process = req.get_json().get('process')
        except ValueError:
            process = 0
    process = int(process)
    if process not in [0, 1, 2, 3, 4] and process < 100:
        process = 0

    char = req.params.get('char')
    if char is None:
        try:
            char = req.get_json().get('char')
        except ValueError:
            pass

    string = req.params.get('string')
    if string is None:
        try:
            string = req.get_json().get('string')
        except ValueError:
            pass
    if string:
        # 文字列をn個結合して長文に変換
        n = 3
        long_string = string
        for _ in range(n-1):
            long_string += " " + string


# Activity関数群
    # process=None または 範囲外 の処理
    if process==0:
        return func.HttpResponse("This HTTP triggered function executed successfully. Process is invalid")
    
    #charを,設定した文字列に置換 (if文でさらに細分化し文字列を設定)
    elif process==1 and char and long_string:  
        if 'a' <= char[0] <= 'i':
            replacement = 'test1'
        elif 'j' <= char[0] <= 's':
            replacement = 'test2'
        else:
            replacement = 'test3'
        long_string = long_string.replace(char, replacement)
        return func.HttpResponse(f"The character {char} was converted => {long_string}.")

    #charを削除        
    elif process==2 and char and long_string:  
        long_string = long_string.replace(char, '')
        return func.HttpResponse(f"The character {char} was delated => {long_string}.")
    
    #charの出現回数
    elif process==3 and char and long_string:  
        char_count = long_string.count(char)
        return func.HttpResponse(f"The character '{char}' appears {char_count} times.")
    
    #回数ソート
    elif process==4 and long_string: 
        words = long_string.split()
        word_counts = Counter(words)
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        result = '\n'.join([f"{word}: {count}" for word, count in sorted_word_counts])
        return func.HttpResponse(result)
    
    # process秒間sleep
    elif process >=100:    
        time.sleep(process)
        return func.HttpResponse(f"It was stopped for {process} seconds.")
    
    else:
        if char is None and string is None:
            return func.HttpResponse(
                "Executed successfully. Pass a char and string in the query stringing or in the request body for a response.", 
                status_code=200 
            )
        if char is None:
            return func.HttpResponse(
                "Executed successfully. Pass a char in the query stringing or in the request body for a response.", 
                status_code=200 
            )
        if string is None:
            return func.HttpResponse(
                "Executed successfully. Pass a string in the query stringing or in the request body for a response.", 
                status_code=200 
            )