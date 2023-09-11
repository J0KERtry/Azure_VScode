import azure.functions as func
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchvision import transforms, datasets
from torchmetrics.functional import accuracy

# データセットのダウンロードと前処理
transform = transforms.Compose([transforms.ToTensor()])
train_val = datasets.MNIST('./', train=True, download=True, transform=transform)
test = datasets.MNIST('./', train=False, download=True, transform=transform)
n_train, n_val = 50000, 10000
torch.manual_seed(0)
train, val = torch.utils.data.random_split(train_val, [n_train, n_val])

batch_size = 256

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

pl.seed_everything(0)

# モデルの定義
class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.Flatten(),  # フラット化レイヤーを追加
            nn.Linear(2352, 10)  # 入力サイズを784に修正
        )

    def forward(self, x):
        return self.net(x)

    # トレーニング、検証、テストのステップを共通化
    def common_step(self, batch, batch_idx, step_name):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        acc = accuracy(y.argmax(dim=-1), t, task='multiclass', num_classes=10, top_k=1)
        self.log(f'{step_name}_loss', loss, on_step=False, on_epoch=True)
        self.log(f'{step_name}_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer




# Azure Functionsルート
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="main", auth_level=func.AuthLevel.ANONYMOUS)
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    net = Net()
    logger = CSVLogger(save_dir='logs', name='my_exp')
    trainer = pl.Trainer(max_epochs=1, deterministic=True, logger=logger)
    trainer.fit(net, train_loader, val_loader)
    results = trainer.test(dataloaders=test_loader)
    
    return func.HttpResponse(str(results))
