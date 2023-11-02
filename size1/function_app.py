## 自由に修正可能なファイル　現在はimageでログを取得するために修正中
import  azure.functions  as  func
import  azure.durable_functions  as  df
from sklearn.datasets import fetch_california_housing  # データセット
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchvision import transforms, datasets
from torchmetrics.functional import accuracy

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
    result = yield context.call_activity("image", '')
    return "Inserted"

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

    # バッチサイズの定義
    batch_size = 256

    # Data Loader を定義
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    # 学習の実行
    net = Net()
    logger = CSVLogger(save_dir='logs', name='my_exp')
    trainer = pl.Trainer(max_epochs=5, deterministic=True, logger=logger)
    trainer.fit(net, train_loader, val_loader)

    # テストデータで評価
    results = trainer.test(dataloaders=test_loader)

    return str(results)

# csvに書き込む関数
@app.blob_output(arg_name="outputblob", path="newblob/size.txt", connection="BlobStorageConnection")
@app.activity_trigger(input_name="size")
def  write_csv(size: int, outputblob: func.Out[str]):
    outputblob.set(str(size))
    return "end"