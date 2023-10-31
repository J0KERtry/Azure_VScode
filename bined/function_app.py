import  azure.functions  as  func
import  azure.durable_functions  as  df
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # 単回帰分析
from sklearn.datasets import fetch_california_housing  # データセット
from sklearn.model_selection import train_test_split  # 分割のためのモジュール
from sklearn.linear_model import Lasso  # Lasso回帰
from sklearn.linear_model import Ridge  # Ridge回帰
from sklearn.metrics import mean_squared_error  # MSE(Mean Squared Error)
from sklearn.preprocessing import StandardScaler  # 標準化ライブラリ
from sklearn.decomposition import PCA  # 主成分分析
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
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)
    return client.create_check_status_response(req, instance_id)

### orchestrator function ###
@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> str:
    result = yield context.call_activity("data", '')
    result = yield context.call_activity("image", '')
    return "Inserted"


### activity function ###
@app.blob_output(arg_name="outputblob", path="newblob/test.txt", connection="BlobStorageConnection")
@app.cosmos_db_output(arg_name="outputDocument", database_name="MyDatabase", container_name="MyCollection", connection="MyAccount_COSMOSDB")
@app.event_grid_output(arg_name="outputEvent", topic_endpoint_uri="MyEventGridTopicUriSetting", topic_key_setting="MyEventGridTopicKeySetting")
@app.activity_trigger(input_name="blank")
def data(blank: str, outputblob: func.Out[str], outputEvent: func.Out[str], outputDocument: func.Out[func.Document]):
    # データの準備
    california_housing = fetch_california_housing()
    # 説明変数
    exp_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    # 目的変数
    tar_data = pd.DataFrame(california_housing.target, columns=['HousingPrices'])
    # データを結合
    data = pd.concat([exp_data, tar_data], axis=1)
    # 異常値の削除
    # 築52年以上のデータ、5.00001以上のデータとしてまとめられている可能性があるため削除
    data = data[data['HouseAge'] != 52]
    # 世帯数、ブロックの全部屋数、ブロックの全寝室数を追加
    data['Household'] = data['Population']/data['AveOccup']
    data['AllRooms'] = data['AveRooms']*data['Household']
    data['AllBedrms'] = data['AveBedrms']*data['Household']

### 単回帰分析 ###
    exp_var = 'MedInc'
    tar_var = 'HousingPrices'

    # 外れ値を除去
    q_95 = data['MedInc'].quantile(0.95)
    # 絞り込む
    data = data[data['MedInc'] < q_95]
    # 説明変数と目的変数にデータを分割
    X = data[[exp_var]]
    y = data[[tar_var]]
    # 学習
    model = LinearRegression()
    model.fit(X, y)

### 重回帰分析 ###
    # 説明変数
    exp_vars = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    # 目的変数
    tar_var = 'HousingPrices'
    # 外れ値を除去
    for exp_var in exp_vars:
        q_95 = data[exp_var].quantile(0.95)
        data = data[data[exp_var] < q_95]
    # 説明変数と目的変数にデータを分割
    X = data[exp_vars]
    y = data[[tar_var]]
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #  X_trainを標準化する
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns = exp_vars)
    # 学習
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    # 予測値を計算
    y_pred = model.predict(X_train_scaled)
    y_pred[:10]
    # テストデータに対する MSE
    X_test_scaled = scaler.transform(X_test) # テストデータを訓練データから得られた平均と標準偏差で標準化
    y_test_pred = model.predict(X_test_scaled) # テストデータに対して予測する
    mse_test = mean_squared_error(y_test, y_test_pred)
    # Ridge回帰
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_y_pred = ridge.predict(X_train_scaled)
    # 偏回帰係数の確認
    ridge_w = pd.DataFrame(ridge.coef_.T, index=exp_vars, columns=['Ridge'])
    for xi, wi in zip(exp_vars, ridge.coef_[0]):
        print('{0:7s}: {1:6.3f}'.format(xi, wi))
    # 訓練データに対する Mean Squared Error (MSE)
    mse_train = mean_squared_error(y_train, y_pred)
    # 訓練データに対する Mean Squared Error (MSE)
    ridge_mse_train = mean_squared_error(y_train, ridge_y_pred)
    # テストデータに対する MSE
    ridge_y_test_pred = ridge.predict(X_test_scaled) # テストデータに対して予測する
    ridge_mse_test = mean_squared_error(y_test, ridge_y_test_pred)
    # Lasso回帰
    lasso = Lasso(alpha=1.0)
    lasso.fit(X_train_scaled, y_train)
    lasso_y_pred = lasso.predict(X_train_scaled)
    # 偏回帰係数の確認
    lasso_w = pd.Series(index=exp_vars, data=lasso.coef_)
    lasso_mse_train = mean_squared_error(y_train, lasso_y_pred)
    lasso_X_test_scaled = scaler.transform(X_test)
    lasso_y_pred_test = lasso.predict(lasso_X_test_scaled)
    lasso_mse_test = mean_squared_error(y_test, lasso_y_pred_test)

    # 正則化ありとなしにおける重回帰分析の精度の比較
    data = {'訓練データMSE':[mse_train, ridge_mse_train, lasso_mse_train],
            'テストデータMSE':[mse_test, ridge_mse_test, lasso_mse_test],
            '決定係数':[model.score(X_test_scaled, y_test), ridge.score(X_test_scaled, y_test), lasso.score(X_test_scaled, y_test)]}
    df_mse = pd.DataFrame(data=data, index=['重回帰', 'Ridge回帰', 'Lasso回帰'])

    return str(df_mse)

@app.blob_output(arg_name="outputblob", path="newblob/test.txt", connection="BlobStorageConnection")
@app.cosmos_db_output(arg_name="outputDocument", database_name="MyDatabase", container_name="MyCollection", connection="MyAccount_COSMOSDB")
@app.event_grid_output(arg_name="outputEvent", topic_endpoint_uri="MyEventGridTopicUriSetting", topic_key_setting="MyEventGridTopicKeySetting")
@app.activity_trigger(input_name="blank")
def image(blank: str, outputblob: func.Out[str], outputEvent: func.Out[str], outputDocument: func.Out[func.Document]):

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