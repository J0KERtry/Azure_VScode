### インスタンスの数 ###

import azure.functions as func
import azure.durable_functions as df
import logging
import numpy as np
import pandas as pd
import pickle
import random
import time

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)
@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async def client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:

    # オーケストレーションの起動
    instance_id = await client.start_new("orchestrator", None, {})
    logging.info(f"Started orchestration with ID = '{instance_id}'.")
    
    # オーケストレーションの完了を待機
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)

    # オーケストレーションの実行状態を取得
    status = await client.get_status(instance_id)

    # オーケストレーションの実行結果を取得
    runtime = status.runtime_status
    output = status.output
    return f"runtime: {runtime}\n\noutput:{output}" 


@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> float:
    start = time.time()
    a = yield context.call_activity("activity1", "")
    time1 = float(time.time() - start)
    b = yield context.call_activity("activity2", "")
    time2 = float(time.time() - (start+time1))
    total_time = float(time.time() - start)
    
    return {"total time" : total_time,
            "time taken for activity1" : time1,
            "time taken for activity2" : time2}


@app.activity_trigger(input_name="blank")
def activity1(blank: str) -> str :
    data = np.random.rand(500*500)
    df = pd.DataFrame(data)
    df_ = df.to_dict()
    return df_

@app.activity_trigger(input_name="blank")
def activity2(blank: str) -> str:
    data = np.random.rand(500*500)
    df = pd.DataFrame(data)
    df_ = df.to_dict()
    return df_
