# 月額料金[円] = { ( リソース使用量[GB/秒] - 400000[GB/秒] ) × 0.001792[円] } + { ( 実行回数[回] - 1000000[回] ) / 1000000 × 22.400[円] }

import azure.functions as func
import logging
import time
from collections import Counter

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
activity_functions = {"failed", "sleep", "replace", "sort", "count_up", "delete"}
SLEEP_TIMES = 10
JOIN_TIMES = 10

def get_param_from_request(req, param_name, json_key=None):
    param = req.params.get(param_name)
    if param is None:
        try:
            param = req.get_json().get(json_key or param_name)
        except (ValueError, KeyError):
            pass
    return param

##################
## クライアント ##
##################
@app.route(route="main")
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # パラメータかjsonから値取得
    process = get_param_from_request(req, "process")
    if process not in activity_functions:
        process = "failed"
    char = get_param_from_request(req, "char")
    string = get_param_from_request(req, "string")

    # 関数呼び出し
    result = execute_process(process, char, string)

    return func.HttpResponse(result)

#########################
## オーケストレーション ##
#########################
def execute_process(process, char, string):
    if process in {"failed", "sleep"}:
        result = globals()[process]()
    else:
        strings = join(string)
        inputs = {"char": char, "strings": strings}
        result = globals()[process](inputs)
        
    return result


#########################
##  アクティビティ関数  ##
#########################
def failed():
    return "failed_function executed successfully."

def sleep():
        time.sleep(SLEEP_TIMES)
        return f"It was stopped for {SLEEP_TIMES} seconds."

# 文字列長変更
def join(string):
    strings = " ".join([string] * JOIN_TIMES)
    return strings

#charを,設定した文字列に置換 (if文でさらに細分化し文字列を設定)
def replace(inputs):
    char = inputs["char"]
    strings = inputs["strings"]

    if 'a' <= char[0] <= 'i':
        replacement = 'test1'
    elif 'j' <= char[0] <= 's':
        replacement = 'test2'
    else:
        replacement = 'test3'

    strings = strings.replace(char, replacement)
    return f"The character {char} was converted => {strings}."

#charを削除        
def delete(inputs):
    char = inputs["char"]
    strings = inputs["strings"]
    strings = strings.replace(char, '')
    return f"The character {char} was delated => {strings}."
    
#charの出現回数
def count_up(inputs):
    char = inputs["char"]
    strings = inputs["strings"]
    char_count = strings.count(char)
    return f"The character '{char}' appears {char_count} times."
    
#回数ソート
def sort(inputs):
        strings = inputs["strings"]
        words = strings.split()
        word_counts = Counter(words)
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        result = '\n'.join([f"{word}: {count}" for word, count in sorted_word_counts])
        return result