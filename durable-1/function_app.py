import azure.functions as func
import azure.durable_functions as df
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchvision import transforms, datasets
from torchmetrics.functional import accuracy


app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)
# -----------------------------------------------------------------------------------------------------------
# クライアント関数 
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


# -----------------------------------------------------------------------------------------------------------
# オーケストレーター関数
@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> str:
    # 前処理
    train_loader, val_loader, test_loader = pre_processing()

    # 学習の実行
    net = Net()
    logger = CSVLogger(save_dir='logs', name='my_exp')
    trainer = pl.Trainer(max_epochs=1, deterministic=True, logger=logger)
    trainer.fit(net, train_loader, val_loader)

    # テストデータで評価
    results = trainer.test(dataloaders=test_loader)

    return str(results)


# -----------------------------------------------------------------------------------------------------------
# アクティビティ関数
def pre_processing():
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
    
    return train_loader, val_loader, test_loader

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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)  # 学習率を調整
        return optimizer

# テストデータでモデルを評価
def evaluation(test_loader, trainer):
    results = trainer.test(dataloaders=test_loader)
    return func.HttpResponse(str(results))