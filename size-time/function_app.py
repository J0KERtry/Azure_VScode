### データサイズと転送方向をsizeとactivityのパラメータで渡し、応答速度が返される ###
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

    # オーケストレーションの実行状態を取得 & 表示
    status = await client.get_status(instance_id)
    return f"runtime: {status.runtime_status}\n\noutput: {status.output}" 


@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> dict:
    parameter = context.get_input()
    activity = int(parameter.get("activity"))
    size = int(parameter.get("size"))
    
    if activity == 1:
        result  =  yield context.call_activity("activity1", size)
        transfer_time  =  time.perf_counter() - result["start"]
        return  {"Transfer_time from act1 to orc": transfer_time}
    
    elif activity == 2:
        data = np.random.rand(size) # データ作成
        df  =  pd.DataFrame(data)
        payload = {"df_": df.to_dict(), "start": time.perf_counter()}
        transfer_time  =  yield context.call_activity("activity2", payload)

        return  {"Transfer_time from orc to act2": transfer_time}


# sizeからDataFrame作成し、アクティビティ関数 -> オーケストレーター関数に転送
@app.activity_trigger(input_name="size")
def  activity1(size: int) -> dict:
    data = np.random.rand(size)  
    df  =  pd.DataFrame(data)
    return {"df_": df.to_dict(), "start": time.perf_counter()}


# オーケストレーション関数 からstartとDataFrame受け取る
@app.activity_trigger(input_name="payload")
def  activity2(payload: dict) -> float:
    start = payload["start"]
    return time.perf_counter() - start