### データの転送 ###

import azure.functions as func
import azure.durable_functions as df
import logging
import numpy as np
import pandas as pd
import time
import pickle


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
def  orchestrator(context: df.DurableOrchestrationContext):
    start  =  time.time()
    result  =  yield context.call_activity("activity1", "")
    transfer_time  =  time.time() -  start
    return  result


@app.activity_trigger(input_name="blank")
def  activity1(blank: str) -> int:
    data = np.random.rand(1024 * 1024 * 5)  # Create random data
    df  =  pd.DataFrame(data) # Data frame creation
    serialized_data = pickle.dumps(df)  # Serialize encode the data
    serialized_size = len(serialized_data)  # Measure the size in bytes
    return serialized_size