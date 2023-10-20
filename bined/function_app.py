import  azure.functions  as  func
import  azure.durable_functions  as  df
import  logging  
import  csv

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
@app.blob_output(arg_name="outputblob",
                path="newblob/test.csv",
                connection="BlobStorageConnection")
@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext, outputblob: func.Out[str]) -> dict:
    string  =  "Data is successfully Inserted"
    logging.info(f'Python Queue trigger function processed {len(string)} bytes')
    outputblob.set(string)
    return "Inserted"

### activity function ###
@app.activity_trigger(input_name="blank")
def main(blank: str, outputblob: func.Out[str]):
    string  =  "Data is successfully Inserted"
    logging.info(f'Python Queue trigger function processed {len(string)} bytes')
    outputblob.set(string)
    return  "Completed"