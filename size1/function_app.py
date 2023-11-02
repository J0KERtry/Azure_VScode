## 自由に修正可能なファイル　現在はimageでログを取得するために修正中
import  azure.functions  as  func
import  azure.durable_functions  as  df
from sklearn.datasets import fetch_california_housing  # データセット
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torchmetrics.functional import accuracy
from azure.storage.blob import BlobServiceClient
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

class AzureBlobLogger(Logger):
    def __init__(self, container_name, blob_name_prefix, connection_string):
        super().__init__()
        self.container_name = container_name
        self.blob_name_prefix = blob_name_prefix
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        if not self.container_client.exists():
            self.container_client.create_container()

    @property
    def name(self):
        return 'AzureBlobLogger'

    @property
    def version(self):
        return '0.0.1'

    @rank_zero_only
    def log_metrics(self, metrics, step):
        blob_name = f"{self.blob_name_prefix}_step_{step}.csv"
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)
        csv_content = ",".join([f"{k},{v}" for k, v in metrics.items()])
        blob_client.upload_blob(csv_content, overwrite=True)

    @rank_zero_only
    def log_hyperparams(self, params):
        # Implement if needed
        pass

class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(588, 10)  # 入力サイズを修正

    def forward(self, x):
        h = self.conv(x)
        h = F.relu(h)
        h = self.bn(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = h.view(-1, 588)
        h = self.fc(h)
        return h
    
    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        train_acc = accuracy(y.argmax(dim=-1), t, task='multiclass', num_classes=10, top_k=1)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        val_acc = accuracy(y.argmax(dim=-1), t, task='multiclass', num_classes=10, top_k=1)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        test_acc = accuracy(y.argmax(dim=-1), t, task='multiclass', num_classes=10, top_k=1)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', test_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)  # 学習率を調整
        return optimizer

app  =  df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)  
@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async  def  client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    instance_id  =  await  client.start_new("orchestrator", None, {})
    await  client.wait_for_completion_or_create_check_status_response(req, instance_id)
    return client.create_check_status_response(req, instance_id)

### orchestrator function ###
@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> str:
    result = yield context.call_activity("image", "")
    return result

### activity function ###
@app.blob_output(arg_name="outputblob", path="newblob/test.txt", connection="BlobStorageConnection")
@app.activity_trigger(input_name="blank")
def image(blank: str, outputblob: func.Out[str]):
    # データセットの変換を定義
    transform = transforms.Compose([transforms.ToTensor()])
    train_val = datasets.MNIST('./', train=True, download=True, transform=transform)
    test = datasets.MNIST('./', train=False, download=True, transform=transform)

    # train と val に分割
    n_train,n_val = 50000, 10000
    torch.manual_seed(0)
    train, val = torch.utils.data.random_split(train_val, [n_train, n_val])

    # Data Loader を定義
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    # 学習の実行
    net = Net()
    azure_logger = AzureBlobLogger(
        container_name="newblob",
        blob_name_prefix="log",
        connection_string="DefaultEndpointsProtocol=https;AccountName=instancedata01;AccountKey=Yor/4Lz9GzT666xMltpyUJzUZDb3SOlExJ13l35MrcKA7Qz7UPdNkL4TECJwP3QVSdceLjJ054WG+AStvp0o/g==;EndpointSuffix=core.windows.net"
    )
    trainer = pl.Trainer(max_epochs=3, deterministic=True, logger=azure_logger)
    trainer.fit(net, train_loader, val_loader)
    results = trainer.test(dataloaders=test_loader)
    return str(results)

# csvに書き込む関数
@app.blob_output(arg_name="outputblob", path="newblob/size.txt", connection="BlobStorageConnection")
@app.activity_trigger(input_name="size")
def  write_csv(size: int, outputblob: func.Out[str]):
    outputblob.set(str(size))
    return "end"