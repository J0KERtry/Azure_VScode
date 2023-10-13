### HTTPパラメータデータフレームのサイズを書き込み、データサイズのみ確認 ###
import azure.functions as func
import azure.durable_functions as df
import logging
import numpy as np
import pandas as pd
import sys


app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)
@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async def client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    try:
        size = int(req.params.get('size') or req.get_json().get('size'))
        instance_id = await client.start_new("orchestrator", None, {"size": size})
        logging.info(f"Started orchestration with ID = '{instance_id}'.")
        await client.wait_for_completion_or_create_check_status_response(req, instance_id)
        status = await client.get_status(instance_id)
        return f"runtime: {status.runtime_status}\n\noutput: {status.output}"
    except Exception as e:
        return func.HttpResponse(str(e), status_code=500)


@app.orchestration_trigger(context_name="context")
def  orchestrator(context: df.DurableOrchestrationContext):
    parameter = context.get_input()
    size = int(parameter.get("size"))

    data = np.random.rand(size)
    df = pd.DataFrame(data)
    df_ = df.to_dict()
    dict_size = sys.getsizeof(df_)
    dict_size += sum(map(sys.getsizeof, df_.values())) + sum(map(sys.getsizeof, df_.keys()))
    
    return  {"transfer_size" : dict_size}