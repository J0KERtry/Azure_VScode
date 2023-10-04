### データの転送 ###

import azure.functions as func
import azure.durable_functions as df
import random
import numpy as np
import pandas as pd
import time
import logging


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
def orchestrator(context: df.DurableOrchestrationContext) -> str:
    start = time.time()
    context.call_activity("activity1", "")
    exec_time = time.time() - start
    return exec_time


@app.activity_trigger(input_name="blank")
def activity1(blank: str):
    data = np.random.rand(1024*1024) # ランダムなデータを生成
    df = pd.DataFrame(data) # DataFrameを作成
    df_json = df.to_json()
    return df_json
