import azure.functions as func
import azure.durable_functions as df
import time
import logging
from collections import Counter

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)
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

############
## Cliant ##
############
@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async def client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    # HTTPリクエストからパラメータを取得
    process = get_param_from_request(req, "process")
    if process not in activity_functions:
        process = "failed"
    char = get_param_from_request(req, "char")
    string = get_param_from_request(req, "string")

    # オーケストレーションの起動
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
    output = status.output
    return f"runtime: {runtime}\n\ninput_:{input_}\n\noutput:{output}" 


##################
## Orchestrator ##
##################
@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> str:
    # クライアント関数から値取得
    parameters = context.get_input()
    process = parameters.get("process")
    char = parameters.get("char")
    string = parameters.get("string")

    # アクティビティ関数の振り分け
    if process in {"failed", "sleep"}:
        result = yield context.call_activity(process, "")
    else:
        strings = yield context.call_activity("join", string)
        inputs = {"char": char, "strings": strings}
        result = yield context.call_activity(process, inputs)
    
    # クライアント関数へ値渡す
    return result


##############
## Activity ##
##############
@app.activity_trigger(input_name="inputs")
def failed(inputs: dict) -> str:
    return "failed_function executed successfully."

@app.activity_trigger(input_name="inputs")
def sleep(inputs: dict) -> str:
    time.sleep(SLEEP_TIMES)
    return f"It was stopped for {SLEEP_TIMES} seconds."

@app.activity_trigger(input_name="string")
def join(string: str) -> str:
    strings = " ".join([string] * JOIN_TIMES)
    return strings

@app.activity_trigger(input_name="inputs")
def replace(inputs: dict) -> str:
    char = inputs["char"]
    strings = inputs["strings"]
    if 'a' <= char[0] <= 'i':
        replacement = 'a~i'
    elif 'j' <= char[0] <= 's':
        replacement = 'j~s'
    else:
        replacement = 't~z'
    strings = strings.replace(char, replacement)
    return f"The character [{char}] was converted => [{strings}]."

@app.activity_trigger(input_name="inputs")
def delete(inputs: dict) -> str:
    char = inputs["char"]
    strings = inputs["strings"]
    strings = strings.replace(char, '')
    return f"The character [{char}] was deleted => [{strings}]."

@app.activity_trigger(input_name="inputs")
def count_up(inputs: dict) -> str:
    char = inputs["char"]
    strings = inputs["strings"]
    char_count = strings.count(char)
    return f"The character [{char}] appears {char_count} times."

@app.activity_trigger(input_name="inputs")
def sort(inputs: dict) -> str:
    strings = inputs["strings"]
    words = strings.split()
    word_counts = Counter(words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    result = '\n'.join([f"{word}: {count}" for word, count in sorted_word_counts])
    return result