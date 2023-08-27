# 月額料金[円] = { ( リソース使用量[GB/秒] - 400000[GB/秒] ) × 0.001792[円] } + { ( 実行回数[回] - 1000000[回] ) / 1000000 × 22.400[円] }

import azure.functions as func
import logging
import time
from collections import Counter

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
activity_functions = {"failed", "sleep", "replace", "sort", "count_up", "delete"}
sleep_times = 10
join_times = 10

@app.route(route="main")
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    ### パラメータかjsonからデータ取得  ###
    process = req.params.get('process')
    if process is None:
        try:
            process = req.get_json().get('process')
        except ValueError:
            pass
    if process not in activity_functions:
        process = "failed"

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

    # 関数呼び出し
    if process == "failed" or process == "sleep":
        result = globals()[process]()
        return func.HttpResponse(result)
    
    strings = join(string)

    inputs = {"char": char, "strings": strings}
    result = globals()[process](inputs)
    return func.HttpResponse(result)
    

##############
##  関数群  ##
##############
def failed():
    return "failed_function executed successfully."

def sleep():
        time.sleep(sleep_times)
        return "It was stopped for {sleep_times} seconds."

# 文字列長変更
def join(string):
    strings = string
    for _ in range(join_times - 1):
        strings += " " + string
    return strings

#charを,設定した文字列に置換 (if文でさらに細分化し文字列を設定)
def replace(inputs):
    char = inputs["char"]
    if 'a' <= char[0] <= 'i':
        replacement = 'test1'
    elif 'j' <= char[0] <= 's':
        replacement = 'test2'
    else:
        replacement = 'test3'
    long_string = long_string.replace(char, replacement)
    return f"The character {char} was converted => {long_string}."

#charを削除        
def delete(inputs):
    char = inputs["char"]
    long_string = inputs["long_string"]
    long_string = long_string.replace(char, '')
    return f"The character {char} was delated => {long_string}."
    
#charの出現回数
def count_up(inputs):
    char = inputs["char"]
    long_string = inputs["long_string"]
    char_count = long_string.count(char)
    return f"The character '{char}' appears {char_count} times."
    
#回数ソート
def sort(inputs):
        long_string = inputs["long_string"]
        words = long_string.split()
        word_counts = Counter(words)
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        result = '\n'.join([f"{word}: {count}" for word, count in sorted_word_counts])
        return result