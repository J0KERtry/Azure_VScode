### データ型とそのサイズを指定し、転送される際のデータサイズを返す ###
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
    
    # Dataframeから辞書型にしたものの転送サイズ
    if activity == 1:
        data = np.random.rand(size)
        df = pd.DataFrame(data)
        df_ = df.to_dict()
        dict_size = sys.getsizeof(df_)
        dict_size += sum(map(sys.getsizeof, df_.values())) + sum(map(sys.getsizeof, df_.keys()))
        return  {"transfer_size(Dataframe->dict)" : dict_size}
    
    # int型の転送サイズ
    # 10MBのint生成-> size = 10 * 1024 * 1024  ※10MBを32ビット整数で表現するための要素数を設定
    elif activity == 2:
        data = np.random.randint(0, 10, size=size // 4, dtype=np.int32)
        data_size = sys.getsizeof(data)
        return {"transfer_size(int)": data_size}
    
    # list型の転送サイズ
    # 10MB のリストを作る-> size = 10*1024*1024
    elif activity == 3:
        data = [0] * (size // sys.getsizeof(0))
        data_size = sys.getsizeof(data)
        return {"transfer_size(list)": data_size}
    
    # 画像
    elif activity == 4:
        return None