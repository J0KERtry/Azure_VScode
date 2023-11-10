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
    ridge = yield context.call_activity("ridge_regression", multiple)
    lasso = yield context.call_activity("lasso_regression", multiple)
    result = yield context.call_activity("evaluate_models", multiple)
    result = yield context.call_activity("origin_analysis", '')
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

    data = data.to_dict()
    return data

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

    model = pickle.dumps(model)
    model = base64.b64encode(model).decode()
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

    model = pickle.dumps(model)
    model = base64.b64encode(model).decode()
    X_train_scaled = pickle.dumps(X_train_scaled)
    X_train_scaled = base64.b64encode(X_train_scaled).decode()
    y_train = pickle.dumps(y_train)
    y_train = base64.b64encode(y_train).decode()
    X_test = pickle.dumps(X_test)
    X_test = base64.b64encode(X_test).decode()
    y_test = pickle.dumps(y_test)
    y_test = base64.b64encode(y_test).decode()
    scaler = pickle.dumps(scaler)
    scaler = base64.b64encode(scaler).decode()
    return {"model": model, "X_train_scaled": X_train_scaled, "y_train": y_train, "X_test": X_test, "y_test": y_test, "scaler": scaler}

# Ridge回帰
@app.activity_trigger(input_name="arg")
def ridge_regression(arg: dict):
    X_train_scaled = pickle.loads(base64.b64decode(arg["X_train_scaled"]))
    y_train = pickle.loads(base64.b64decode(arg["y_train"]))

    # Ridge回帰
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)

    ridge = pickle.dumps(ridge)
    ridge = base64.b64encode(ridge).decode()
    return ridge

# Lasso回帰
@app.activity_trigger(input_name="arg")
def lasso_regression(arg: dict):
    X_train_scaled = pickle.loads(base64.b64decode(arg["X_train_scaled"]))
    y_train = pickle.loads(base64.b64decode(arg["y_train"]))

    # Lasso回帰
    lasso = Lasso(alpha=1.0)
    lasso.fit(X_train_scaled, y_train)

    lasso = pickle.dumps(lasso)
    lasso = base64.b64encode(lasso).decode()
    return lasso

# 評価
@app.activity_trigger(input_name="arg")
def evaluate_models(arg: dict):
    model = pickle.loads(base64.b64decode(arg["model"]))
    X_train_scaled = pickle.loads(base64.b64decode(arg["X_train_scaled"]))
    y_train = pickle.loads(base64.b64decode(arg["y_train"]))
    X_test = pickle.loads(base64.b64decode(arg["X_test"]))
    y_test = pickle.loads(base64.b64decode(arg["y_test"]))
    scaler = pickle.loads(base64.b64decode(arg["scaler"]))

    mse_train = []
    mse_test = []
    r2 = []

    # 予測値を計算
    y_pred = model.predict(X_train_scaled)
    # 訓練データに対する Mean Squared Error (MSE)
    mse_train.append(mean_squared_error(y_train, y_pred))
    # テストデータに対する MSE
    X_test_scaled = scaler.transform(X_test) # テストデータを訓練データから得られた平均と標準偏差で標準化
    y_test_pred = model.predict(X_test_scaled) # テストデータに対して予測する
    mse_test.append(mean_squared_error(y_test, y_test_pred))
    # 決定係数
    r2.append(model.score(X_test_scaled, y_test))

    # 正則化ありとなしにおける重回帰分析の精度の比較
    data = {'訓練データMSE': mse_train,
            'テストデータMSE': mse_test,
            '決定係数': r2}
    df_mse = pd.DataFrame(data=data, index=['重回帰', 'Ridge回帰', 'Lasso回帰'])

    df_mse = base64.b64encode(pickle.dumps(df_mse)).decode()
    return df_mse


#################################################
# 元のモノリシックコード
@app.activity_trigger(input_name="blank")
def origin_analysis(blank: str):

    california_housing = fetch_california_housing() # データの準備

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