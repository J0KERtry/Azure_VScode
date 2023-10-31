### 1~12個のインスタンスの数をactivityパラメータで渡し、各インスタンスで作成するDataFrameのサイズをsizeで渡す ###
import azure.functions as func
import azure.durable_functions as df

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)
@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async def client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    try:
        activity = int(req.params.get('activity') or req.get_json().get('activity'))
    except Exception as e:
        return func.HttpResponse("Invalid 'activity' or 'size' parameters.", status_code=400)
    instance_id = await client.start_new("orchestrator", None, {"activity": activity})
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)
    return client.create_check_status_response(req, instance_id)

@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> dict:
    parameter = context.get_input()
    activity = int(parameter.get("activity"))
    for i in range(1, activity + 1):
        activity_func = context.call_activity(f"activity{i}", '')
    return 'end'


# Azure Cosmos Database, Azure Blob Strage, Azure Event Grid
@app.blob_output(arg_name="outputblob",
                path="newblob/result.txt",
                connection="BlobStorageConnection")
@app.cosmos_db_output(arg_name="outputDocument",
                      database_name="MyDatabase",
                      container_name="MyCollection",
                      connection="MyAccount_COSMOSDB")
@app.event_grid_output(arg_name="outputEvent",
                       topic_endpoint_uri="MyEventGridTopicUriSetting",
                       topic_key_setting="MyEventGridTopicKeySetting")
@app.activity_trigger(input_name="blank")
def  activity1(blank: str,
          outputblob: func.Out[str],
          outputEvent: func.Out[str],
          outputDocument: func.Out[func.Document]):
    
    return  "OK"

@app.blob_output(arg_name="outputblob",
                path="newblob/result.txt",
                connection="BlobStorageConnection")
@app.cosmos_db_output(arg_name="outputDocument",
                      database_name="MyDatabase",
                      container_name="MyCollection",
                      connection="MyAccount_COSMOSDB")
@app.event_grid_output(arg_name="outputEvent",
                       topic_endpoint_uri="MyEventGridTopicUriSetting",
                       topic_key_setting="MyEventGridTopicKeySetting")
@app.activity_trigger(input_name="blank")
def  activity2(blank: str,
          outputblob: func.Out[str],
          outputEvent: func.Out[str],
          outputDocument: func.Out[func.Document]):
    return  "OK"

@app.blob_output(arg_name="outputblob",
                path="newblob/result.txt",
                connection="BlobStorageConnection")
@app.cosmos_db_output(arg_name="outputDocument",
                      database_name="MyDatabase",
                      container_name="MyCollection",
                      connection="MyAccount_COSMOSDB")
@app.event_grid_output(arg_name="outputEvent",
                       topic_endpoint_uri="MyEventGridTopicUriSetting",
                       topic_key_setting="MyEventGridTopicKeySetting")
@app.activity_trigger(input_name="blank")
def  activity3(blank: str,
          outputblob: func.Out[str],
          outputEvent: func.Out[str],
          outputDocument: func.Out[func.Document]):
    return  "OK"

@app.blob_output(arg_name="outputblob",
                path="newblob/result.txt",
                connection="BlobStorageConnection")
@app.cosmos_db_output(arg_name="outputDocument",
                      database_name="MyDatabase",
                      container_name="MyCollection",
                      connection="MyAccount_COSMOSDB")
@app.event_grid_output(arg_name="outputEvent",
                       topic_endpoint_uri="MyEventGridTopicUriSetting",
                       topic_key_setting="MyEventGridTopicKeySetting")
@app.activity_trigger(input_name="blank")
def  activity4(blank: str,
          outputblob: func.Out[str],
          outputEvent: func.Out[str],
          outputDocument: func.Out[func.Document]):
    return  "OK"

@app.blob_output(arg_name="outputblob",
                path="newblob/result.txt",
                connection="BlobStorageConnection")
@app.cosmos_db_output(arg_name="outputDocument",
                      database_name="MyDatabase",
                      container_name="MyCollection",
                      connection="MyAccount_COSMOSDB")
@app.event_grid_output(arg_name="outputEvent",
                       topic_endpoint_uri="MyEventGridTopicUriSetting",
                       topic_key_setting="MyEventGridTopicKeySetting")
@app.activity_trigger(input_name="blank")
def  activity5(blank: str,
          outputblob: func.Out[str],
          outputEvent: func.Out[str],
          outputDocument: func.Out[func.Document]):
    return  "OK"

@app.blob_output(arg_name="outputblob",
                path="newblob/result.txt",
                connection="BlobStorageConnection")
@app.cosmos_db_output(arg_name="outputDocument",
                      database_name="MyDatabase",
                      container_name="MyCollection",
                      connection="MyAccount_COSMOSDB")
@app.event_grid_output(arg_name="outputEvent",
                       topic_endpoint_uri="MyEventGridTopicUriSetting",
                       topic_key_setting="MyEventGridTopicKeySetting")
@app.activity_trigger(input_name="blank")
def  activity6(blank: str,
          outputblob: func.Out[str],
          outputEvent: func.Out[str],
          outputDocument: func.Out[func.Document]):
    return  "OK"

@app.blob_output(arg_name="outputblob",
                path="newblob/result.txt",
                connection="BlobStorageConnection")
@app.cosmos_db_output(arg_name="outputDocument",
                      database_name="MyDatabase",
                      container_name="MyCollection",
                      connection="MyAccount_COSMOSDB")
@app.event_grid_output(arg_name="outputEvent",
                       topic_endpoint_uri="MyEventGridTopicUriSetting",
                       topic_key_setting="MyEventGridTopicKeySetting")
@app.activity_trigger(input_name="blank")
def  activity7(blank: str,
          outputblob: func.Out[str],
          outputEvent: func.Out[str],
          outputDocument: func.Out[func.Document]):
    return  "OK"

@app.blob_output(arg_name="outputblob",
                path="newblob/result.txt",
                connection="BlobStorageConnection")
@app.cosmos_db_output(arg_name="outputDocument",
                      database_name="MyDatabase",
                      container_name="MyCollection",
                      connection="MyAccount_COSMOSDB")
@app.event_grid_output(arg_name="outputEvent",
                       topic_endpoint_uri="MyEventGridTopicUriSetting",
                       topic_key_setting="MyEventGridTopicKeySetting")
@app.activity_trigger(input_name="blank")
def  activity8(blank: str,
          outputblob: func.Out[str],
          outputEvent: func.Out[str],
          outputDocument: func.Out[func.Document]):
    return  "OK"

@app.blob_output(arg_name="outputblob",
                path="newblob/result.txt",
                connection="BlobStorageConnection")
@app.cosmos_db_output(arg_name="outputDocument",
                      database_name="MyDatabase",
                      container_name="MyCollection",
                      connection="MyAccount_COSMOSDB")
@app.event_grid_output(arg_name="outputEvent",
                       topic_endpoint_uri="MyEventGridTopicUriSetting",
                       topic_key_setting="MyEventGridTopicKeySetting")
@app.activity_trigger(input_name="blank")
def  activity9(blank: str,
          outputblob: func.Out[str],
          outputEvent: func.Out[str],
          outputDocument: func.Out[func.Document]):
    return  "OK"

@app.blob_output(arg_name="outputblob",
                path="newblob/result.txt",
                connection="BlobStorageConnection")
@app.cosmos_db_output(arg_name="outputDocument",
                      database_name="MyDatabase",
                      container_name="MyCollection",
                      connection="MyAccount_COSMOSDB")
@app.event_grid_output(arg_name="outputEvent",
                       topic_endpoint_uri="MyEventGridTopicUriSetting",
                       topic_key_setting="MyEventGridTopicKeySetting")
@app.activity_trigger(input_name="blank")
def  activity10(blank: str,
          outputblob: func.Out[str],
          outputEvent: func.Out[str],
          outputDocument: func.Out[func.Document]):
    return  "OK"