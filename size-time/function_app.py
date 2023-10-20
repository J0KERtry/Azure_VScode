### データサイズと型をsizeとactivityのパラメータで渡し、応答速度が返される ###
import azure.functions as func
import azure.durable_functions as df
import logging
import numpy as np
import pandas as pd
import time
import sys
import csv

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
    
    if activity == 1: #DataFrame
        result  =  yield context.call_activity("activity1", size)
        data_frame = pd.DataFrame.from_dict(result["data"])  # デシリアライズ
        transfer_time  =  time.perf_counter() - result["start"] # デシリアライズが終了したら転送時間記録

        dict_size = sys.getsizeof(result["data"]) # 転送時間の計測が終わったら、データサイズを計測
        dict_size += sum(map(sys.getsizeof, result["data"].values())) + sum(map(sys.getsizeof, result["data"].keys()))
        output = {"DataFrame": size, "datasize": dict_size, "Transfer_time": transfer_time}

        with open('test.csv', 'a', newline='') as f:
            header = ['DataFrame', 'datasize', 'Transfer_time']
            writer = csv.DictWriter(f, header)
            writer.writerow({'DataFrame': size, 'datasize': dict_size, 'Transfer_time': transfer_time})
    
    elif activity == 2: # Numpy配列
        result  =  yield context.call_activity("activity2", size)
        receive = np.array(result["data"])
        transfer_time  =  time.perf_counter() - result["start"] # 転送データを受け取ったら転送時間記録
        output = {"DataFrame": size, "datasize": result["data_size"], "Transfer_time": transfer_time}

        with open('test.csv', 'a', newline='') as f:
            header = ['DataFrame', 'datasize', 'Transfer_time']
            writer = csv.DictWriter(f, header)
            writer.writerow({'DataFrame': size, 'datasize': result['data_size'], 'Transfer_time': transfer_time})
    
    elif activity == 3: # list
        result  =  yield context.call_activity("activity3", size)
        receive = result["data"]
        transfer_time  =  time.perf_counter() - result["start"]
        output = {"DataFrame": size, "datasize": result["data_size"], "Transfer_time": transfer_time}

        with open('test.csv', 'a', newline='') as f:
            header = ['DataFrame', 'datasize', 'Transfer_time']
            writer = csv.DictWriter(f, header)
            writer.writerow({'DataFrame': size, 'datasize': result['data_size'], 'Transfer_time': transfer_time})

    return output


# DataFrameを作成し転送
@app.activity_trigger(input_name="size")
def  activity1(size: int) -> dict:
    data = np.random.rand(size) # 1行size列のNumpy配列作成
    data  =  pd.DataFrame(data) # Numpy配列をDataframeに変換
    start = time.perf_counter() # Dataframeをシリアライズ可能な辞書型に変換する時間も測るため、ここに表記
    data_ = data.to_dict() # シリアライズ可能な型に変換
    return {"data": data_, "start": start}

# Numpy配列を作成し転送
# 10MBのデータ生成-> size = 10*1024*1024
@app.activity_trigger(input_name="size")
def  activity2(size: int) -> dict:
    data = np.random.randint(0, 10, size=size // 4, dtype=np.int32)
    data_size = sys.getsizeof(data)
    start = time.perf_counter()
    data = data.tolist()
    return {"data": data, "data_size": data_size, "start": start}

# リスト型を作成し転送
# 10MB のリストを作る-> size = 10*1024*1024
@app.activity_trigger(input_name="size")
def  activity3(size: int) -> dict:
    data = [0] * (size // sys.getsizeof(0))
    data_size = sys.getsizeof(data)
    start = time.perf_counter()
    return {"data": data, "data_size": data_size, "start": start}
