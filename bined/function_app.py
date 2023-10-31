import  azure.functions  as  func
import  azure.durable_functions  as  df
import  logging  
import  numpy as np
import  pandas as pd

app  =  df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)  

### client function ###
@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async  def  client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    instance_id  =  await  client.start_new("orchestrator", None, {})
    logging.info(f"Started orchestration with ID = '{instance_id}'.")
    await  client.wait_for_completion_or_create_check_status_response(req, instance_id)
    status  =  await  client.get_status(instance_id)
    return  f"output: {status.output}"  

### orchestrator function ###
@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> str:
    result = yield context.call_activity("main", '')
    return "Inserted"

### activity function ###
@app.blob_output(arg_name="outputblob", path="newblob/test.csv", connection="BlobStorageConnection")
@app.activity_trigger(input_name="blank")
def main(blank: str, outputblob: func.Out[str]):
    list1 = [100, 200, 300, 400, 0]
    list2 = [0.1, 0.2, 0.3, 0.4, 0.5]
    output = {"DataFrame": list1, "Transfer_time": list2}
    df = pd.DataFrame(output)
    csv_data = df.to_csv(index=False)
    outputblob.set(csv_data)
    return "Inserted"