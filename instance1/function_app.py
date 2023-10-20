### 1~12個のインスタンスの数をactivityパラメータで渡し、各インスタンスで作成するDataFrameのサイズをsizeで渡す ###
import azure.functions as func
import azure.durable_functions as df
import logging
import numpy as np
import pandas as pd
import time

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async def client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    try:
        activity = int(req.params.get('activity') or req.get_json().get('activity'))
        size = int(req.params.get('size') or req.get_json().get('size'))
    except Exception as e:
        return func.HttpResponse("Invalid 'activity' or 'size' parameters.", status_code=400)

    instance_id = await client.start_new("orchestrator", None, {"activity": activity, "size": size})
    logging.info(f"Started orchestration with ID = '{instance_id}'.")
    
    # オーケストレーションの完了を待機
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)

    # オーケストレーションの実行状態を取得
    status = await client.get_status(instance_id)
    output_str = "\n\n".join([f"{key}: {value}" for key, value in status.output.items()])
    return f"runtime: {status.runtime_status}\n\noutput:\n {output_str}" 


@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> dict:
    parameter = context.get_input()
    activity = int(parameter.get("activity"))
    size = int(parameter.get("size"))
    
    total_time = 0.0
    times = {}  # 各アクティビティの実行時間を記録
    start = time.perf_counter()
    for i in range(1, activity + 1):
        activity_func = context.call_activity(f"activity{i}", size)
        time_taken = float(time.perf_counter() - (start + total_time))
        times[f"time taken for activity{i}"] = time_taken
        total_time += time_taken
    
    times["total time"] = total_time
    return times


@app.activity_trigger(input_name="size")
def activity1(size: int) -> int:
    data = np.random.rand(size) 
    df  =  pd.DataFrame(data) 
    df_ = df.to_dict()
    return len(df_)

@app.activity_trigger(input_name="size")
def activity2(size: int) -> int:
    data = np.random.rand(size) 
    df  =  pd.DataFrame(data)
    df_ = df.to_dict()
    return len(df_)

@app.activity_trigger(input_name="size")
def activity3(size: int) -> int:
    data = np.random.rand(size) 
    df  =  pd.DataFrame(data)
    df_ = df.to_dict()
    return len(df_)

@app.activity_trigger(input_name="size")
def activity4(size: int) -> int:
    data = np.random.rand(size) 
    df  =  pd.DataFrame(data) 
    df_ = df.to_dict()
    return len(df_)

@app.activity_trigger(input_name="size")
def activity5(size: int) -> int:
    data = np.random.rand(size) 
    df  =  pd.DataFrame(data)
    df_ = df.to_dict()
    return len(df_)

@app.activity_trigger(input_name="size")
def activity6(size: int) -> int:
    data = np.random.rand(size) 
    df  =  pd.DataFrame(data)
    df_ = df.to_dict()
    return len(df_)

@app.activity_trigger(input_name="size")
def activity7(size: int) -> int:
    data = np.random.rand(size) 
    df  =  pd.DataFrame(data) 
    df_ = df.to_dict()
    return len(df_)

@app.activity_trigger(input_name="size")
def activity8(size: int) -> int:
    data = np.random.rand(size) 
    df  =  pd.DataFrame(data)
    df_ = df.to_dict()
    return len(df_)

@app.activity_trigger(input_name="size")
def activity9(size: int) -> int:
    data = np.random.rand(size) 
    df  =  pd.DataFrame(data)
    df_ = df.to_dict()
    return len(df_)

@app.activity_trigger(input_name="size")
def activity10(size: int) -> int:
    data = np.random.rand(size) 
    df  =  pd.DataFrame(data) 
    df_ = df.to_dict()
    return len(df_)

@app.activity_trigger(input_name="size")
def activity11(size: int) -> int:
    data = np.random.rand(size) 
    df  =  pd.DataFrame(data)
    df_ = df.to_dict()
    return len(df_)

@app.activity_trigger(input_name="size")
def activity12(size: int) -> int:
    data = np.random.rand(size) 
    df  =  pd.DataFrame(data)
    df_ = df.to_dict()
    return len(df_)