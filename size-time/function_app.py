### データサイズと型をsizeとactivityのパラメータで渡し、応答速度が返される ###
import azure.functions as func
import azure.durable_functions as df
import numpy as np
import pandas as pd
import pickle
import base64

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)
@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async def client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    try:
        size = int(req.params.get('size') or req.get_json().get('size'))
    except Exception as e:
        return func.HttpResponse("Invalid 'activity' or 'size' parameters.", status_code=400)
    
    instance_id = await client.start_new("orchestrator", None, {"size": size})
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)  # オーケストレーションの完了を待機
    return client.create_check_status_response(req, instance_id)


@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> dict:
    parameter = context.get_input()
    size = int(parameter.get("size"))

    for i in range(size):
        result  =  yield context.call_activity("activity1", size)
        data = pickle.loads(base64.b64decode(result["data"]))

    return 'orchestrator end'


# Numpy配列を作成し転送
@app.activity_trigger(input_name="size")
def  activity1(size: int) -> dict:
    data = np.random.randint(0, 100, size= 196610 * 10, dtype=np.int32)
    data = base64.b64encode(pickle.dumps(data)).decode()
    return { "data": data }