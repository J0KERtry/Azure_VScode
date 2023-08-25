import azure.functions as func
import azure.durable_functions as df
import time
import logging
import json
from collections import Counter

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

############
## Cliant ##
############
@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async def client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    # HTTPリクエストから必要なパラメータを取得
    process = req.params.get("process")
    if process is None:
        try:
            process = req.get_json().get('process')
        except (ValueError, KeyError):
            process = 0
    process = int(process)
    if process not in [0, 1, 2, 3, 4, 5] and process < 100:
        process = 0

    char = req.params.get('char')
    if char is None:
        try:
            char = req.get_json().get('char')
        except (ValueError, KeyError):
            pass

    string = req.params.get('string')
    if string is None:
        try:
            string = req.get_json().get('string')
        except (ValueError, KeyError):
            pass

    instance_id = await client.start_new("orchestrator", None, {
        "process": process,
        "char": char,
        "string": string
    })
    logging.info(f"Started orchestration with ID = '{instance_id}'.")
    
    # オーケストレーションの完了を待機
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)

    # オーケストレーションの実行状態を取得
    status = await client.get_status(instance_id)

    # オーケストレーションの実行結果を取得
    runtime = status.runtime_status
    input_ = status.input_
    output = status.output  ## オーケストレーターインスタンスの完了によって返される JSON シリアル化可能な値を取得
    history = status.history
    return f"runtime: {runtime}\n\ninput_:{input_}\n\noutput:{output}\n\nhistory:{history}" 


##################
## Orchestrator ##
##################
@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> str:
    parameters = context.get_input()
    process = parameters.get("process")
    char = parameters.get("char")
    string = parameters.get("string")

    result = None
    # アクティビティ関数の条件分岐
    # process=0 のこの部分を修正して正しい出力を得れればよい
    if process == 0:
        result = context.call_activity("failed") # 返り値の型：<class 'azure.durable_functions.models.Task.AtomicTask'>
        result = str(result)
        context.set_custom_status(result) # TypeError: Object of type AtomicTask is not JSON serializable
        return result

    long_string = context.call_activity("join", string)
    if process == 1:
        result = context.call_activity("replace", char, long_string)
    elif process == 2:
        result = context.call_activity("delete", char, long_string)
    elif process == 3:
        result = context.call_activity("count_up", char, long_string)
    elif process == 4:
        result = context.call_activity("sort", long_string)
    elif process == 5:
        result = context.call_activity("sleep", process)
    else:
        if char is None and string is None:
            result = "Executed successfully. Pass a char and string in the query or in the request body."
        if char is None:
            result = "Executed successfully. Pass a char in the query or in the request body."
        if string is None:
            result = "Executed successfully. Pass a string in the query or in the request body."

    context.set_custom_status(result)
    return result

##############
## Activity ##
##############
@app.activity_trigger
def failed() -> str:
    return "failed_function executed successfully."

@app.activity_trigger
def join(string: str) -> str:
    n = 5
    long_string = string
    for _ in range(n-1):
        long_string += " " + string
    return long_string

@app.activity_trigger
def replace(char: str, long_string: str) -> str:
    if 'a' <= char[0] <= 'i':
        replacement = 'a~i'
    elif 'j' <= char[0] <= 's':
        replacement = 'j~s'
    else:
        replacement = 't~z'
    long_string = long_string.replace(char, replacement)
    return f"The character [{char}] was converted => [{long_string}]."

@app.activity_trigger
def delete(char: str, long_string: str) -> str:
    long_string = long_string.replace(char, '')
    return f"The character [{char}] was deleted => [{long_string}]."

@app.activity_trigger
def count_up(char: str, long_string: str) -> str:
    char_count = long_string.count(char)
    return f"The character [{char}] appears {char_count} times."

@app.activity_trigger
def sort(long_string: str) -> str:
    words = long_string.split()
    word_counts = Counter(words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    result = '\n'.join([f"{word}: {count}" for word, count in sorted_word_counts])
    return result

@app.activity_trigger
def sleep(process: int) -> str:
    time.sleep(process)
    return f"It was stopped for {process} seconds."