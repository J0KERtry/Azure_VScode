import azure.functions as func
import azure.durable_functions as df
import time
from collections import Counter

app = df.DurableOrchestrationClient()

############
## Cliant ##
############
@app.function("DurableClientExample")
async def durable_client(parameters: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    client = df.DurableOrchestrationClient(context)
    function_name = parameters.route_params.get('functionName')
    instance_id = await client.start_new(function_name)
    return client.create_check_status_response(parameters, instance_id)


##################
## Orchestrator ##
##################
@df.func
@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext):
    parameters = context.get_input()  # HTTPリクエストのパラメータを受け取る
    
    # パラメータの値を取得
    process = parameters.get("process")
    if process is None:
        try:
            process = parameters.get_json().get('process')
        except (ValueError, KeyError):
            process = 0
    process = int(process)
    if process not in [0, 1, 2, 3, 4] and process < 100:
        process = 0

    char = parameters.get('char')
    if char is None:
        try:
            char = parameters.get_json().get('char')
        except (ValueError, KeyError):
            pass

    string = parameters.get('string')
    if string is None:
        try:
            string = parameters.get_json().get('string')
        except (ValueError, KeyError):
            pass

    # アクティビティ関数の条件分岐
    if process == 0:
        return func.HttpResponse("This HTTP triggered function executed successfully. Process is invalid")
    long_string = context.call_activity("join", string)
    if process == 1:
        context.call_activity("replace", char, long_string)
    elif process == 2:
        context.call_activity("delete", char, long_string)
    elif process == 3:
        context.call_activity("count_up", char, long_string)
    elif process == 4:
        context.call_activity("sort", long_string)
    else:
        if char is None and string is None:
            return func.HttpResponse(
                "Executed successfully. Pass a char and string in the query stringing or in the parametersuest body.", 
                status_code=200 
            )
        if char is None:
            return func.HttpResponse(
                "Executed successfully. Pass a char in the query stringing or in the parametersuest body.", 
                status_code=200 
            )
        if string is None:
            return func.HttpResponse(
                "Executed successfully. Pass a string in the query stringing or in the parametersuest body.", 
                status_code=200 
            )


##############
## Activity ##
##############
@app.activity_trigger(input_name="input_value")
def join(long_string: str):
    n = 5
    for _ in range(n-1):
        long_string += " " + long_string
    return long_string

@app.activity_trigger(input_name="input_value")
def replace(char: str, long_string:str):
    if 'a' <= char[0] <= 'i':
        replacement = 'a~i'
    elif 'j' <= char[0] <= 's':
        replacement = 'j~s'
    else:
        replacement = 't~z'
    long_string = long_string.replace(char, replacement)
    return func.HttpResponse(f"The character [{char}] was converted => [{long_string}].")

@app.activity_trigger(input_name="input_value")
def delete(char: str, long_string:str):
    long_string = long_string.replace(char, '')
    return func.HttpResponse(f"The character [{char}] was delated => [{long_string}].")

@app.activity_trigger(input_name="input_value")
def count_up(char: str, long_string:str):
    char_count = long_string.count(char)
    return func.HttpResponse(f"The character [{char}] appears {char_count} times.")

@app.activity_trigger(input_name="input_value")
def sort(long_string: str):
    words = long_string.split()
    word_counts = Counter(words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    result = '\n'.join([f"{word}: {count}" for word, count in sorted_word_counts])
    return func.HttpResponse(result)

@app.activity_trigger(input_name="input_value")
def sleep(process: int):
    time.sleep(process)
    return func.HttpResponse(f"It was stopped for {process} seconds.")        