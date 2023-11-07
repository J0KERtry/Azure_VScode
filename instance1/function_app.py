# データ分析、機械学習のコードを両方記述
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

app  =  df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)  
@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async  def  client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    split = int(req.params.get('split') or req.get_json().get('split'))
    instance_id  =  await  client.start_new("orchestrator", None, {"split":split})
    await  client.wait_for_completion_or_create_check_status_response(req, instance_id)
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)
    return client.create_check_status_response(req, instance_id)

### orchestrator function ###
@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> str:
    parameter = context.get_input()
    split = int(parameter.get("split"))

    result = yield context.call_activity("a_code", '')
    result = yield context.call_activity("loop", 2)
    
    for i in range(split):
        result = yield context.call_activity("a_code", '')
    
    result = yield context.call_activity("loop", split)
    
    return "Inserted"


### activity function ###
@app.blob_output(arg_name="outputblob", path="newblob/test.txt", connection="BlobStorageConnection")
@app.activity_trigger(input_name="split")
def loop(split: int, outputblob: func.Out[str]):
    for i in range(split):
        california_housing = fetch_california_housing()
        exp_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
        tar_data = pd.DataFrame(california_housing.target, columns=['HousingPrices'])
        data = pd.concat([exp_data, tar_data], axis=1)
        data = data[data['HouseAge'] != 52]
        data['Household'] = data['Population']/data['AveOccup']
        data['AllRooms'] = data['AveRooms']*data['Household']
        data['AllBedrms'] = data['AveBedrms']*data['Household']
        exp_var = 'MedInc'
        tar_var = 'HousingPrices'
        q_95 = data['MedInc'].quantile(0.95)
        data = data[data['MedInc'] < q_95]
        X = data[[exp_var]]
        y = data[[tar_var]]
        model = LinearRegression()
        model.fit(X, y)
        exp_vars = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        tar_var = 'HousingPrices'
        for exp_var in exp_vars:
            q_95 = data[exp_var].quantile(0.95)
            data = data[data[exp_var] < q_95]
        X = data[exp_vars]
        y = data[[tar_var]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns = exp_vars)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_train_scaled)
        y_pred[:10]
        X_test_scaled = scaler.transform(X_test) # テストデータを訓練データから得られた平均と標準偏差で標準化
        y_test_pred = model.predict(X_test_scaled) # テストデータに対して予測する
        mse_test = mean_squared_error(y_test, y_test_pred)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        ridge_y_pred = ridge.predict(X_train_scaled)
        ridge_w = pd.DataFrame(ridge.coef_.T, index=exp_vars, columns=['Ridge'])
        for xi, wi in zip(exp_vars, ridge.coef_[0]):
            print('{0:7s}: {1:6.3f}'.format(xi, wi))
        mse_train = mean_squared_error(y_train, y_pred)
        ridge_mse_train = mean_squared_error(y_train, ridge_y_pred)
        ridge_y_test_pred = ridge.predict(X_test_scaled) 
        ridge_mse_test = mean_squared_error(y_test, ridge_y_test_pred)
        lasso = Lasso(alpha=1.0)
        lasso.fit(X_train_scaled, y_train)
        lasso_y_pred = lasso.predict(X_train_scaled)
        lasso_w = pd.Series(index=exp_vars, data=lasso.coef_)
        lasso_mse_train = mean_squared_error(y_train, lasso_y_pred)
        lasso_X_test_scaled = scaler.transform(X_test)
        lasso_y_pred_test = lasso.predict(lasso_X_test_scaled)
        lasso_mse_test = mean_squared_error(y_test, lasso_y_pred_test)
        data = {'訓練データMSE':[mse_train, ridge_mse_train, lasso_mse_train],
                'テストデータMSE':[mse_test, ridge_mse_test, lasso_mse_test],
                '決定係数':[model.score(X_test_scaled, y_test), ridge.score(X_test_scaled, y_test), lasso.score(X_test_scaled, y_test)]}
        df_mse = pd.DataFrame(data=data, index=['重回帰', 'Ridge回帰', 'Lasso回帰'])
    return str(df_mse)

@app.blob_output(arg_name="outputblob", path="newblob/test.txt", connection="BlobStorageConnection")
@app.activity_trigger(input_name="blank")
def a_code(blank: str, outputblob: func.Out[str]):
    california_housing = fetch_california_housing()
    exp_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    tar_data = pd.DataFrame(california_housing.target, columns=['HousingPrices'])
    data = pd.concat([exp_data, tar_data], axis=1)
    data = data[data['HouseAge'] != 52]
    data['Household'] = data['Population']/data['AveOccup']
    data['AllRooms'] = data['AveRooms']*data['Household']
    data['AllBedrms'] = data['AveBedrms']*data['Household']
    exp_var = 'MedInc'
    tar_var = 'HousingPrices'
    q_95 = data['MedInc'].quantile(0.95)
    data = data[data['MedInc'] < q_95]
    X = data[[exp_var]]
    y = data[[tar_var]]
    model = LinearRegression()
    model.fit(X, y)
    exp_vars = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    tar_var = 'HousingPrices'
    for exp_var in exp_vars:
        q_95 = data[exp_var].quantile(0.95)
        data = data[data[exp_var] < q_95]
    X = data[exp_vars]
    y = data[[tar_var]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns = exp_vars)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_train_scaled)
    y_pred[:10]
    X_test_scaled = scaler.transform(X_test) # テストデータを訓練データから得られた平均と標準偏差で標準化
    y_test_pred = model.predict(X_test_scaled) # テストデータに対して予測する
    mse_test = mean_squared_error(y_test, y_test_pred)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_y_pred = ridge.predict(X_train_scaled)
    ridge_w = pd.DataFrame(ridge.coef_.T, index=exp_vars, columns=['Ridge'])
    for xi, wi in zip(exp_vars, ridge.coef_[0]):
        print('{0:7s}: {1:6.3f}'.format(xi, wi))
    mse_train = mean_squared_error(y_train, y_pred)
    ridge_mse_train = mean_squared_error(y_train, ridge_y_pred)
    ridge_y_test_pred = ridge.predict(X_test_scaled) 
    ridge_mse_test = mean_squared_error(y_test, ridge_y_test_pred)
    lasso = Lasso(alpha=1.0)
    lasso.fit(X_train_scaled, y_train)
    lasso_y_pred = lasso.predict(X_train_scaled)
    lasso_w = pd.Series(index=exp_vars, data=lasso.coef_)
    lasso_mse_train = mean_squared_error(y_train, lasso_y_pred)
    lasso_X_test_scaled = scaler.transform(X_test)
    lasso_y_pred_test = lasso.predict(lasso_X_test_scaled)
    lasso_mse_test = mean_squared_error(y_test, lasso_y_pred_test)
    data = {'訓練データMSE':[mse_train, ridge_mse_train, lasso_mse_train],
            'テストデータMSE':[mse_test, ridge_mse_test, lasso_mse_test],
            '決定係数':[model.score(X_test_scaled, y_test), ridge.score(X_test_scaled, y_test), lasso.score(X_test_scaled, y_test)]}
    df_mse = pd.DataFrame(data=data, index=['重回帰', 'Ridge回帰', 'Lasso回帰'])

    return str(df_mse)