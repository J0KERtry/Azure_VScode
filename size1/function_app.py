### データの転送 ###

import azure.functions as func
import azure.durable_functions as df
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
    input_ = status.input_
    output = status.output
    return f"runtime: {runtime}\n\ninput_:{input_}\n\noutput:{output}" 


@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> str:
    result = yield context.call_activity("activity1", "")
    return result


@app.activity_trigger(input_name="blank")
def activity1(blank: str):
    data_1kb = bytearray(1024)  # 1KBのデータを生成
    x = 3
    y = 4
    return data_1kb

data_1kb = bytearray(1024)  # 1KBのデータを生成
data_1mb = bytearray(1024 * 1024)  # 1MBのデータを生成
data_1gb = bytearray(1024 * 1024 * 1024)  # 1GBのデータを生成
