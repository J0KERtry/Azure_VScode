#### データ分析。機能別に実装。 ####
import azure.functions as func
import azure.durable_functions as df
import pandas as pd
from sklearn.linear_model import LinearRegression  # 単回帰分析
from sklearn.datasets import fetch_california_housing  # データセット
from sklearn.model_selection import train_test_split  # 分割のためのモジュール
from sklearn.linear_model import Lasso  # Lasso回帰
from sklearn.linear_model import Ridge  # Ridge回帰
from sklearn.metrics import mean_squared_error  # MSE(Mean Squared Error)
from sklearn.preprocessing import StandardScaler  # 標準化ライブラリ
import pickle
import base64

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)
### クライアント関数 ###
@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async def client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    instance_id = await client.start_new("orchestrator", None, {})
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)
    return client.create_check_status_response(req, instance_id)

### オーケストレーター関数 ###
@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> str:
    data = yield context.call_activity("prepare_data", '')
    simple = yield context.call_activity("simple_regression", {"data": data})
    multiple = yield context.call_activity("multiple_regression", {"data": data})
    # result = yield context.call_activity("origin_analysis", '')
    return "finished"


### アクティビティ関数 ###
# 前処理
@app.activity_trigger(input_name="blank")
def prepare_data(blank: str):
    # データの準備
    california_housing = fetch_california_housing()
    exp_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names) # 説明変数
    tar_data = pd.DataFrame(california_housing.target, columns=['HousingPrices']) # 目的変数
    data = pd.concat([exp_data, tar_data], axis=1) # データを結合

    # 異常値の削除
    # 築52年以上のデータ、5.00001以上のデータとしてまとめられている可能性があるため削除
    data = data[data['HouseAge'] != 52]
    data = data[data['HousingPrices'] != 5.00001]

    # 世帯数、ブロックの全部屋数、ブロックの全寝室数を追加
    data['Household'] = data['Population']/data['AveOccup']
    data['AllRooms'] = data['AveRooms']*data['Household']
    data['AllBedrms'] = data['AveBedrms']*data['Household']

    return data.to_dict()

# 単回帰分析
@app.activity_trigger(input_name="arg")
def simple_regression(arg: dict):
    data = pd.DataFrame.from_dict(arg['data'])
    exp_var = 'MedInc'
    tar_var = 'HousingPrices'

    # 外れ値を除去
    q_95 = data[exp_var].quantile(0.95)
    data = data[data[exp_var] < q_95]

    # 説明変数と目的変数にデータを分割
    X = data[[exp_var]]
    y = data[[tar_var]]

    # 学習
    model = LinearRegression()
    model.fit(X, y)

    model = base64.b64encode(pickle.dumps(model)).decode()
    return model

# 重回帰分析
@app.activity_trigger(input_name="arg")
def multiple_regression(arg: dict):
    data = pd.DataFrame.from_dict(arg['data'])
    exp_vars = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'] # 説明変数
    tar_var = 'HousingPrices' # 目的変数

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

    # 訓練データに対する Mean Squared Error (MSE)
    mse_train = mean_squared_error(y_train, y_pred)

    # テストデータに対する MSE
    X_test_scaled = scaler.transform(X_test) # テストデータを訓練データから得られた平均と標準偏差で標準化
    y_test_pred = model.predict(X_test_scaled) # テストデータに対して予測する
    mse_test = mean_squared_error(y_test, y_test_pred)


### Ridge回帰 ###
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_y_pred = ridge.predict(X_train_scaled)
    
    # 訓練データに対する Mean Squared Error (MSE)
    ridge_mse_train = mean_squared_error(y_train, ridge_y_pred)
    # テストデータに対する Mean Squared Error (MSE)
    ridge_y_test_pred = ridge.predict(X_test_scaled) # テストデータに対して予測する
    ridge_mse_test = mean_squared_error(y_test, ridge_y_test_pred)

    ridge = base64.b64encode(pickle.dumps(ridge)).decode()


#### Lasso回帰 ###
    lasso = Lasso(alpha=1.0)
    lasso.fit(X_train_scaled, y_train)
    lasso_y_pred = lasso.predict(X_train_scaled)

    # 訓練データに対する Mean Squared Error (MSE)
    lasso_mse_train = mean_squared_error(y_train, lasso_y_pred)
    # テストデータに対する Mean Squared Error (MSE)
    lasso_y_test_pred = lasso.predict(X_test_scaled)
    lasso_mse_test = mean_squared_error(y_test, lasso_y_test_pred)

    lasso = base64.b64encode(pickle.dumps(lasso)).decode()
    return ridge, lasso


#################################################
# 元のモノリシックコード
@app.activity_trigger(input_name="blank")
def origin_analysis(blank: str):
# データの準備
    california_housing = fetch_california_housing() 
    exp_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names) # 説明変数
    tar_data = pd.DataFrame(california_housing.target, columns=['HousingPrices']) # 目的変数
    data = pd.concat([exp_data, tar_data], axis=1) # データを結合

    # 異常値の削除
    data = data[data['HouseAge'] != 52] # 築52年以上のデータ
    data = data[data['HousingPrices'] != 5.00001] # 5.00001以上のデータとしてまとめられている可能性があるため削除

    # 世帯数、ブロックの全部屋数、ブロックの全寝室数を追加
    data['Household'] = data['Population']/data['AveOccup']
    data['AllRooms'] = data['AveRooms']*data['Household']
    data['AllBedrms'] = data['AveBedrms']*data['Household']


### 単回帰分析 ###
    exp_var = 'MedInc'
    tar_var = 'HousingPrices'

    q_95 = data['MedInc'].quantile(0.95) # 外れ値を除去
    data = data[data['MedInc'] < q_95] # 絞り込む

    # 説明変数と目的変数にデータを分割
    X = data[[exp_var]]
    y = data[[tar_var]]

    # 学習
    model = LinearRegression()
    model.fit(X, y)


### 重回帰分析 ###
    exp_vars = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'] # 説明変数
    tar_var = 'HousingPrices' # 目的変数

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

    # 訓練データに対する Mean Squared Error (MSE)
    mse_train = mean_squared_error(y_train, y_pred)

    # テストデータに対する MSE
    X_test_scaled = scaler.transform(X_test) # テストデータを訓練データから得られた平均と標準偏差で標準化
    y_test_pred = model.predict(X_test_scaled) # テストデータに対して予測する
    mse_test = mean_squared_error(y_test, y_test_pred)


### Ridge回帰 ###
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_y_pred = ridge.predict(X_train_scaled)

    # 訓練データに対する Mean Squared Error (MSE)
    ridge_mse_train = mean_squared_error(y_train, ridge_y_pred)
    # テストデータに対する Mean Squared Error (MSE)
    ridge_y_test_pred = ridge.predict(X_test_scaled) # テストデータに対して予測する
    ridge_mse_test = mean_squared_error(y_test, ridge_y_test_pred)


### Lasso回帰 ###
    lasso = Lasso(alpha=1.0)
    lasso.fit(X_train_scaled, y_train)
    lasso_y_pred = lasso.predict(X_train_scaled)

    # 訓練データに対する Mean Squared Error (MSE)
    lasso_mse_train = mean_squared_error(y_train, lasso_y_pred)
    # テストデータに対する Mean Squared Error (MSE)
    lasso_y_test_pred = lasso.predict(X_test_scaled)
    lasso_mse_test = mean_squared_error(y_test, lasso_y_test_pred)

    return 0