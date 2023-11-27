''' 
Github(https://github.com/plotly-dash-apps/502-california-housing-regression/blob/main/analysis/california-housing-simplified.ipynb)
を参考にデータ分析 
'''
import azure.functions as func
import azure.durable_functions as df
import numpy as np
import pandas as pd
import matplotlib as mpl
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.datasets import fetch_california_housing  # データセット
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
import seaborn as sns
import plotly.express as px

import pickle
import base64
import time

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)
### クライアント関数 ###
@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async def client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    instance_id = await client.start_new("orchestrator", None, {})
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)
    status = await client.get_status(instance_id)
    return f"output:{status.output}" 

### オーケストレーター関数 ###
@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> str:
    time = []
    housing = yield context.call_activity("exploratory_data_analysis", '')  # housing = housing.to_dict()
    time.append(housing["time"])

    data = yield context.call_activity("data_preprocessing", housing["housing"])   # data = {std_scaler, train, test...}
    time.append(data["time"])

    multi = yield context.call_activity("multivariate_linear_regression", data)     # multi = {rmse_r2 coinflip, ols}
    time.append(multi["time"])

    param = {"X_train_scaled": data["X_train_scaled"], "X_test_scaled": data["X_test_scaled"], 
             "y_train": data["y_train"], "y_test": data["y_test"]}
    ridge = yield context.call_activity("ridge_regression", param) 
    time.append(ridge["time"])

    knn = yield context.call_activity("k_nearest_neighbor", param) 
    time.append(knn["time"])

    tree = yield context.call_activity("decision_tree_regression", param) 
    time.append(tree["time"])

    forest = yield context.call_activity("random_forest", param) 
    time.append(forest["time"])

    result = yield context.call_activity("result_visualization", {**multi, **ridge, **knn, **tree, **forest})
    time.append(result["time"]) 
    return time

#################################################
### 探索的データ分析 ###
@app.blob_input(arg_name="inputblob", path="dataset/housing.csv", connection="BlobStorageConnection")
@app.activity_trigger(input_name="blank")
def exploratory_data_analysis(blank: str, inputblob: func.InputStream) -> dict:
    start = time.perf_counter()

    # 出力を再現可能にするためにシードを設定
    np.random.seed(42)
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)

    # データセットの読み込み
    housing = pd.read_csv(inputblob)
    housing.sample(5)
    housing['median_house_value'].describe()
    housing['latitude'].describe()
    housing['longitude'].describe()
    housing['median_income'].describe()
    housing['housing_median_age'].describe()
    housing['total_rooms'].describe()
    housing['population'].describe()
    housing['households'].describe()

    # 収入カテゴリの作成
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    housing["income_cat"].value_counts().sort_index(ascending=True)

    # いくつかの追加変数の作成
    housing["rooms_per_hhold"] = housing["total_rooms"]/housing["households"]
    housing["pop_per_household"]=housing["population"]/housing["households"]
    housing["rooms_per_hhold"].describe()
    housing["pop_per_household"].describe()
    housing['ocean_proximity'].value_counts().sort_values(ascending=False)

    # オーシャンの近接度のダミー変数の作成（ワンホットエンコーディング）
    housing=pd.get_dummies(housing, columns=['ocean_proximity'], prefix='', prefix_sep='')

    # 変数の一覧
    housing.columns
    housing.head()

    end = time.perf_counter() - start
    return {"housing": housing.to_dict(), "time": end}


### データの前処理 ###
@app.activity_trigger(input_name="input")
def data_preprocessing(input: dict):
    housing = pd.DataFrame.from_dict(input)

    start = time.perf_counter()
    # sklearnは欠損データを処理できないため、欠損値を削除
    print("len(housing):",len(housing))
    print("housing.isnull: ",housing.isnull().sum())
    housing.drop('total_bedrooms', axis=1, inplace=True)
    print("len(housing):", len(housing))

    # データを特徴量（X）とラベル（y）に分割
    y = housing["median_house_value"].copy()
    # 'ocean_proximity'のカテゴリのうち1つを削除して、係数が解釈可能になるようにする
    X = housing.drop(["median_house_value", '<1H OCEAN'], axis=1)
    # 形状を確認
    print("y.shape:", y.shape)
    print("X.shape:", X.shape)
    X.columns

    # Xとyデータをトレーニングセットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 形状が予定通りになったことを確認するために形状を比較
    print("X_train.shape:", X_train.shape)
    print("y_train.shape:", y_train.shape)
    print("X_test.shape:", X_test.shape)
    print("y_test.shape:", y_test.shape)

    # データセットの数値変数
    X_train.describe().columns

    # X_trainデータにスケーラーを「fit」させる
    std_scaler = StandardScaler()
    std_scaler = std_scaler.fit(X_train.values)

    # スケーラーを使用してデータセットを変換
    X_train_scaled = std_scaler.transform(X_train.values)
    X_test_scaled = std_scaler.transform(X_test.values)

    end = time.perf_counter() - start

    std_scaler = base64.b64encode(pickle.dumps(std_scaler)).decode()
    X_train = base64.b64encode(pickle.dumps(X_train)).decode()
    X_train_scaled = base64.b64encode(pickle.dumps(X_train_scaled)).decode()
    X_test = base64.b64encode(pickle.dumps(X_test)).decode()
    X_test_scaled = base64.b64encode(pickle.dumps(X_test_scaled)).decode()
    y_train = base64.b64encode(pickle.dumps(y_train)).decode()
    y_test = base64.b64encode(pickle.dumps(y_test)).decode()
    data = {
        "std_scaler": std_scaler,
        "X_train": X_train, "X_train_scaled": X_train_scaled, "X_test": X_test, "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "time": end
    }
    return data


### 多変量線形回帰分析 ###
@app.activity_trigger(input_name="input")
def multivariate_linear_regression(input: dict):
    std_scaler = pickle.loads(base64.b64decode(input['std_scaler']))
    X_train = pickle.loads(base64.b64decode(input['X_train']))
    X_train_scaled = pickle.loads(base64.b64decode(input['X_train_scaled']))
    X_test = pickle.loads(base64.b64decode(input['X_test']))
    X_test_scaled = pickle.loads(base64.b64decode(input['X_test_scaled']))
    y_train = pickle.loads(base64.b64decode(input['y_train']))
    y_test = pickle.loads(base64.b64decode(input['y_test']))

    start = time.perf_counter()

    # 線形回帰モデルを作成
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(X_train_scaled, y_train)
    # 切片と係数を確認
    print("lin_reg.intercept_:", lin_reg.intercept_)
    print("lin_reg.coef_:", lin_reg.coef_)

    # 'Attributes' は、特徴（予測変数、独立変数）のリストの別名
    attributes=X_test.columns
    print("attributes:", attributes)
    # 'Feature importances' は、係数の別名（つまり、各特徴が成果または DV に与える影響）
    feature_importances=lin_reg.coef_
    print("feature_importances:", feature_importances)
    # 2つの要素は同じ長さ
    print("len(feature_importances):", len(feature_importances))
    print("len(attributes):", len(attributes))
    # 係数を整数に変換
    feature_importances = [int(x) for x in feature_importances] 

    # 結果
    feature_imp = pd.DataFrame(list(zip(attributes, feature_importances)), columns=['features', 'coeffs'])
    feature_imp=feature_imp.set_index('features')
    feature_imp=feature_imp.sort_values('coeffs')

    # 棒グラフをプロット
    feature_imp.plot(kind='bar');
    data = go.Bar(x=list(feature_imp.index), y=feature_imp['coeffs'])
    coefs = go.Figure([data])

    # 単一の観測の変数を表示
    print("X_train.iloc[0]: ", X_train.iloc[0])

    # 単一の観測のスケールされた変数を表示
    # 変数 "<1H OCEAN" を削除したことに注意
    print("X_train_scaled[0]:", X_train_scaled[0])

    # 予測
    lin_reg.predict([X_train_scaled[0]])

    # 似たような偽のデータを作成
    fake = np.array([-122, 37, 40, 2000, 3000, 500, 3, 3, 6, 4, 0, 0, 1, 0]).reshape(1, -1)

    # 訓練されたスケーラーを使用して標準化
    std_fake = std_scaler.transform(fake)

    # 予測を試す
    lin_reg.predict(std_fake)

    # テストデータセットの予測を行う
    y_preds = lin_reg.predict(X_test_scaled)

    # 最初の5つの予測と、最初の5つの実際の値を比較
    true_5=list(round(y_test[:5], 1))
    pred_5=[round(x,1) for x in list(y_preds[:5])]
    print('true values:', true_5)
    print('predicted values:', pred_5)
    # 結果をどのように解釈するか
    first_5=['district0', 'district1', 'district2', 'district3', 'distict4']
    pd.DataFrame(list(zip(first_5, true_5, pred_5)), columns=['district', 'true', 'predicted'])

    # ルート平均二乗誤差（RMSE）は、モデルの平均誤差（ドル）を表します
    rmse_ols = np.sqrt(metrics.mean_squared_error(y_test, y_preds))
    rmse_ols = int(rmse_ols)

    # コイン投げ（つまり、トレーニングセットの平均）と比較して、どれほど優れていますか？
    avg_val = round(y_train.mean(),2)

    # それを予測値として使用した場合、モデルの平均誤差（RMSE）
    coinflip_preds=np.full((len(y_test), ), avg_val)
    rmse_coinflip=np.sqrt(metrics.mean_squared_error(y_test, coinflip_preds))
    rmse_coinflip=int(rmse_coinflip)

    # R-squaredは、DVの分散のうちモデルによって説明される割合
    r2_ols=metrics.r2_score(y_test, y_preds)
    r2_ols=round(r2_ols, 2)

    # コイン投げ（つまり、トレーニングセットの平均）と比較して、どれほど優れていますか？
    r2_coinflip=metrics.r2_score(y_test, coinflip_preds)
    r2_coinflip=round(r2_coinflip,2)

    # OLS線形回帰をベースラインと比較
    evaluation_df = pd.DataFrame([['Baseline',rmse_coinflip, r2_coinflip],
                                      ['OLS Linear Regression', rmse_ols, r2_ols]],
                                     columns=['Model','RMSE','R-squared']
                                    )
    evaluation_df.set_index('Model', inplace=True)

    # RMSE: Plotlyを使用した棒グラフ
    trace = go.Bar(x=list(evaluation_df.index), y=evaluation_df['RMSE'], marker=dict(color=['#E53712', '#1247E5']))
    layout = go.Layout(title = '地区ごとの平均家屋価値：ルート平均二乗誤差', # グラフのタイトル
            yaxis = dict(title = 'モデル'), # x軸ラベル
            xaxis = dict(title = 'RMSE'), # y軸ラベル
                          )
    fig = go.Figure(data = [trace], layout=layout)

    # R-squared: Plotlyを使用した棒グラフ
    trace = go.Bar(x=list(evaluation_df.index), y=evaluation_df['R-squared'], marker=dict(color=['#E53712', '#1247E5']))
    layout = go.Layout(title = '地区ごとの平均家屋価値：R-squared', # グラフのタイトル
            yaxis = dict(title = 'モデル'), # x軸ラベル
            xaxis = dict(title = 'R-Squared'), # y軸ラベル
                          )
    fig = go.Figure(data = [trace], layout=layout)

    # データ可視化
    fig = sns.regplot(x=y_preds, y=y_test)

    # Plotlyでも可視化
    fig = px.scatter(y=y_test, x=y_preds, trendline="ols", width=500, height=500)
    fig.update_layout(title='地区別の住宅の中間値',  # グラフタイトル
                      yaxis=dict(title='真の値'),  # x軸ラベル
                      xaxis=dict(title='予測値'),  # y軸ラベル
                      )
    fig.update_traces(line_color='#E53712', line_width=5)
    fig.show()

    end = time.perf_counter() - start

    rmse_coinflip = base64.b64encode(pickle.dumps(rmse_coinflip)).decode()
    r2_coinflip = base64.b64encode(pickle.dumps(r2_coinflip)).decode()
    rmse_ols = base64.b64encode(pickle.dumps(rmse_ols)).decode()
    r2_ols = base64.b64encode(pickle.dumps(r2_ols)).decode()
    return {"rmse_coinflip": rmse_coinflip, "r2_coinflip": r2_coinflip, "rmse_ols": rmse_ols, "r2_ols": r2_ols, "time": end }


### Ridge回帰モデル ###
@app.activity_trigger(input_name="input")
def ridge_regression(input: dict):
    X_train_scaled = pickle.loads(base64.b64decode(input['X_train_scaled']))
    X_test_scaled = pickle.loads(base64.b64decode(input['X_test_scaled']))
    y_train = pickle.loads(base64.b64decode(input['y_train']))
    y_test = pickle.loads(base64.b64decode(input['y_test']))

    start = time.perf_counter()

    # Ridge回帰モデルを訓練データに当てはめる
    ridge_model = linear_model.Ridge(alpha=.5)
    ridge_model.fit(X_train_scaled, y_train)

    # テストデータに対する予測
    y_preds = ridge_model.predict(X_test_scaled)

    # 平均二乗誤差（RMSE）は、モデルの平均誤差（ドル単位）を表す
    ridge_rmse = int(np.sqrt(metrics.mean_squared_error(y_test, y_preds)))

    # R2スコアは、モデルによって説明されるDVの分散の割合です
    ridge_r2 = round(metrics.r2_score(y_test, y_preds), 2)
    print(ridge_rmse, ridge_r2)

    end = time.perf_counter() - start

    ridge_rmse = base64.b64encode(pickle.dumps(ridge_rmse)).decode()
    ridge_r2 = base64.b64encode(pickle.dumps(ridge_r2)).decode()
    return {"ridge_rmse": ridge_rmse, "ridge_r2": ridge_r2, "time": end}


### K近傍法モデル ###
@app.activity_trigger(input_name="input")
def k_nearest_neighbor(input: dict):
    X_train_scaled = pickle.loads(base64.b64decode(input['X_train_scaled']))
    X_test_scaled = pickle.loads(base64.b64decode(input['X_test_scaled']))
    y_train = pickle.loads(base64.b64decode(input['y_train']))
    y_test = pickle.loads(base64.b64decode(input['y_test']))

    start = time.perf_counter()

    # K近傍法モデルを訓練データに当てはめる
    knn_model = KNeighborsRegressor(n_neighbors=8)
    knn_model.fit(X_train_scaled, y_train)

    # テストデータに対する予測
    y_preds = knn_model.predict(X_test_scaled)

    # 平均二乗誤差（RMSE）は、モデルの平均誤差（ドル単位）を表す
    knn_rmse = int(np.sqrt(metrics.mean_squared_error(y_test, y_preds)))

    # R2スコアは、モデルによって説明されるDVの分散の割合
    knn_r2 = round(metrics.r2_score(y_test, y_preds), 2)
    print(knn_rmse, knn_r2)

    end = time.perf_counter() - start

    knn_rmse = base64.b64encode(pickle.dumps(knn_rmse)).decode()
    knn_r2 = base64.b64encode(pickle.dumps(knn_r2)).decode()
    return {"knn_rmse": knn_rmse, "knn_r2": knn_r2, "time": end}


### 決定木回帰モデル ###
@app.activity_trigger(input_name="input")
def decision_tree_regression(input: dict):
    X_train_scaled = pickle.loads(base64.b64decode(input['X_train_scaled']))
    X_test_scaled = pickle.loads(base64.b64decode(input['X_test_scaled']))
    y_train = pickle.loads(base64.b64decode(input['y_train']))
    y_test = pickle.loads(base64.b64decode(input['y_test']))

    start = time.perf_counter()

    # 決定木回帰モデルを訓練データに当てはめる
    tree_model = DecisionTreeRegressor(max_depth=9)
    tree_model.fit(X_train_scaled, y_train)

    # テストデータに対する予測
    y_preds = tree_model.predict(X_test_scaled)

    # 平均二乗誤差（RMSE）は、モデルの平均誤差（ドル単位）を表す
    tree_rmse = int(np.sqrt(metrics.mean_squared_error(y_test, y_preds)))

    # R2スコアは、モデルによって説明されるDVの分散の割合
    tree_r2 = round(metrics.r2_score(y_test, y_preds), 2)
    print(tree_rmse, tree_r2)

    end = time.perf_counter() - start

    tree_rmse = base64.b64encode(pickle.dumps(tree_rmse)).decode()
    tree_r2 = base64.b64encode(pickle.dumps(tree_r2)).decode()
    return {"tree_rmse": tree_rmse, "tree_r2": tree_r2, "time": end}


### random forestモデル ###
@app.activity_trigger(input_name="input")
def random_forest(input: dict):
    X_train_scaled = pickle.loads(base64.b64decode(input['X_train_scaled']))
    X_test_scaled = pickle.loads(base64.b64decode(input['X_test_scaled']))
    y_train = pickle.loads(base64.b64decode(input['y_train']))
    y_test = pickle.loads(base64.b64decode(input['y_test']))

    start = time.perf_counter()

    # random forestモデルを訓練データに当てはめる
    forest_model = RandomForestRegressor(max_depth=10, n_estimators=200)
    forest_model.fit(X_train_scaled, y_train)

    # テストデータセットで予測
    y_preds = forest_model.predict(X_test_scaled)

    # ルート平均二乗誤差（RMSE）は、モデルの平均誤差（ドル単位）を表す
    forest_rmse = int(np.sqrt(metrics.mean_squared_error(y_test, y_preds)))

    # R2スコアは、説明変数によって説明される従属変数の分散の割合
    forest_r2=round(metrics.r2_score(y_test, y_preds),2)

    print(forest_rmse, forest_r2)

    end = time.perf_counter() - start

    forest_rmse = base64.b64encode(pickle.dumps(forest_rmse)).decode()
    forest_r2 = base64.b64encode(pickle.dumps(forest_r2)).decode()
    return {"forest_rmse": forest_rmse, "forest_r2": forest_r2, "time": end}


### 棒グラフで結果可視化 ###
@app.activity_trigger(input_name="input")
def result_visualization(input: dict):
    rmse_coinflip = pickle.loads(base64.b64decode(input['rmse_coinflip']))
    r2_coinflip = pickle.loads(base64.b64decode(input['r2_coinflip']))
    rmse_ols = pickle.loads(base64.b64decode(input['rmse_ols']))
    r2_ols = pickle.loads(base64.b64decode(input['r2_ols']))
    ridge_rmse = pickle.loads(base64.b64decode(input['ridge_rmse']))
    ridge_r2 = pickle.loads(base64.b64decode(input['ridge_r2']))
    knn_rmse = pickle.loads(base64.b64decode(input['knn_rmse']))
    knn_r2 = pickle.loads(base64.b64decode(input['ridge_r2']))
    tree_rmse = pickle.loads(base64.b64decode(input['tree_rmse']))
    tree_r2 = pickle.loads(base64.b64decode(input['tree_r2']))
    forest_rmse = pickle.loads(base64.b64decode(input['forest_rmse']))
    forest_r2 = pickle.loads(base64.b64decode(input['forest_r2']))

    start = time.perf_counter()

    evaluation_df2 = pd.DataFrame([['ベースライン',rmse_coinflip, r2_coinflip],
                                      ['OLS線形回帰', rmse_ols, r2_ols],
                                      ['リッジ回帰', ridge_rmse, ridge_r2],
                                      ['K近傍法回帰', knn_rmse, knn_r2],
                                      ['決定木回帰', tree_rmse, tree_r2],
                                      ['ランダムフォレスト回帰', forest_rmse, forest_r2]],
                                     columns=['モデル','RMSE','R-squared']
                                    )
    evaluation_df2.set_index('モデル', inplace=True)

    # RMSEの棒グラフを作成
    trace = go.Bar(x=list(evaluation_df2.index),
                       y=evaluation_df2['RMSE'],
                       marker=dict(color=['gray', '#e96060', 'gray', 'gray', 'gray', 'gray']),
                      )
    layout = go.Layout(title = 'モデル比較: ルート平均二乗誤差', # グラフタイトル
            yaxis = dict(title = 'モデル'), # x軸ラベル
            xaxis = dict(title = 'RMSE'), # y軸ラベル
                          )
    rmse_fig = go.Figure(data = [trace], layout=layout)

    # R2スコアの棒グラフを作成
    trace = go.Bar(x=list(evaluation_df2.index),
                       y=evaluation_df2['R-squared'],
                       marker=dict(color=['gray', '#e96060', 'gray', 'gray', 'gray', 'gray']),
                      )
    layout = go.Layout(title = 'モデル比較: R-Squared', # グラフタイトル
            yaxis = dict(title = 'モデル'), # x軸ラベル
            xaxis = dict(title = 'R-Squared'), # y軸ラベル
                          )
    r2_fig = go.Figure(data = [trace], layout=layout)

    end = time.perf_counter() - start

    evaluation_df2 = base64.b64encode(pickle.dumps(evaluation_df2)).decode()
    rmse_fig = base64.b64encode(pickle.dumps(rmse_fig)).decode()
    r2_fig = base64.b64encode(pickle.dumps(r2_fig)).decode()
    return {"evaluation_df2": evaluation_df2, "rmse_fig": rmse_fig, "r2_fig": r2_fig, "time": end}
