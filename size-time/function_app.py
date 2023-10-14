### データサイズをコードで設定し、応答速度を計測 ###
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
    activity = int(req.params.get('activity') or req.get_json().get('activity'))
    size = int(req.params.get('size') or req.get_json().get('size'))

    instance_id = await client.start_new("orchestrator", None, {"activity":activity, "size":size})
    logging.info(f"Started orchestration with ID = '{instance_id}'.")
    
    # オーケストレーションの完了を待機
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)

    # オーケストレーションの実行状態を取得 & 表示
    status = await client.get_status(instance_id)
    return f"runtime: {status.runtime_status}\n\noutput: {status.output}" 


@app.orchestration_trigger(context_name="context")
def  orchestrator(context: df.DurableOrchestrationContext):
    parameter = context.get_input()
    activity = int(parameter.get("activity"))
    size = int(parameter.get("size"))
    
    if activity == 1:
        result  =  yield context.call_activity("activity1", "size")
        start = result("start")
        transfer_time  =  time.time() - start
        return  {"transfer_time" : transfer_time}
    
    elif activity == 2:
        data = np.random.rand(size) 
        df  =  pd.DataFrame(data) # DataFrame作成
        df_ = df.to_dict() # DataFrameをシリアライズ可能なdictに変換
        start  =  time.time()
        result  =  yield context.call_activity("activity2", "start")
        transfer_time = result("transfer_time")
        return  {"transfer_time" : transfer_time}


# sizeからDataFrame作成し、アクティビティ関数 -> オーケストレーター関数に転送
@app.activity_trigger(input_name="blank")
def  activity1(context: df.DurableOrchestrationContext) -> str:
    parameter = context.get_input()
    size = int(parameter.get("size"))

    data = np.random.rand(size)  
    df  =  pd.DataFrame(data) # DataFrame作成
    df_ = df.to_dict() # DataFrameをシリアライズ可能なdictに変換
    return df_


# オーケストレーション関数 からstartとDataFrame受け取る
@app.activity_trigger(input_name="blank")
def  activity2(context: df.DurableOrchestrationContext) -> float:
    parameter = context.get_input()
    start = int(parameter.get("start"))
    transfer_time = time.time() - start
    return transfer_time