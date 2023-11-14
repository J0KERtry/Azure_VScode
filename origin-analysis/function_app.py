#### データ分析。機能別に実装。 ####
import azure.functions as func
import azure.durable_functions as df
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    result = yield context.call_activity("origin_analysis", '')
    return "finished"

#################################################
# 元のモノリシックコード
@app.activity_trigger(input_name="blank")
def origin_analysis(blank: str):
    # 出力を再現可能にするためにシードを設定
    np.random.seed(42)
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)

    # データセットの読み込み
    housing = pd.read_csv("housing.csv")
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

    # sklearnは欠損データを処理できないため、欠損値を削除
    print(len(housing))
    print(housing.isnull().sum())
    housing.dropna(axis=1, inplace=True)
    print(len(housing))

    # データを特徴量（X）とラベル（y）に分割
    y = housing["median_house_value"].copy()
    # 'ocean_proximity'のカテゴリのうち1つを削除して、係数が解釈可能になるようにする
    X = housing.drop(["median_house_value", '<1H OCEAN'], axis=1)
    # 形状を確認
    print(y.shape)
    print(X.shape)

    X.columns

    # さらに、Xとyデータをトレーニングセットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    # 形状が予定通りになったことを確認するために形状を比較
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # データセットの数値変数は何ですか？
    X_train.describe().columns

    # スケーラーの「インスタンス化」（sklearnクラスのインスタンスを作成）
    std_scaler = StandardScaler()

    # X_trainデータにスケーラーを「適合」させる
    std_scaler = std_scaler.fit(X_train.values)

    # スケーラーを使用してデータセットを変換
    X_train_scaled = std_scaler.transform(X_train.values)
    X_train_scaled[0]

    # スケーラーを使用してデータセットを変換
    X_test_scaled = std_scaler.transform(X_test.values)
    X_test_scaled[0]

    # sklearnクラスのローカルインスタンスを作成
    lin_reg = LinearRegression(fit_intercept=True)

    # トレーニングデータセットにインスタンスを適合させる
    lin_reg.fit(X_train_scaled, y_train)

    # 切片と係数を確認
    print(lin_reg.intercept_)
    print(lin_reg.coef_)

    # 'Attributes'は特徴量のリストの別名です（予測子、独立変数）
    attributes = X_test.columns
    print(attributes)
    # 'Feature importances'は係数の別名です（各特徴量が結果またはDVに与える影響）
    feature_importances = lin_reg.coef_
    print(feature_importances)

    # 明らかに、これら2つのものは同じ長さを持っているでしょう
    print(len(feature_importances))
    print(len(attributes))

    [int(x) for x in list(feature_importances)]

    # 結果を見てみましょう
    feature_imp = pd.DataFrame(list(zip(attributes, feature_importances)), columns=['features', 'coeffs'])
    feature_imp = feature_imp.set_index('features')
    feature_imp = feature_imp.sort_values('coeffs')

    # 棒グラフとしてプロット
    feature_imp.plot(kind='bar');

    # plotlyを使用して同じことを行う
    data = go.Bar(x=list(feature_imp.index), y=feature_imp['coeffs'])
    coefs = go.Figure([data])

    X_train.columns

    # 単一の観測の変数を表示
    print(X_train.iloc[0])

    # その単一の観測のスケーリングされた変数を表示
    # '<1H OCEAN'変数を削除したことを覚えておいてください
    print(X_train_scaled[0])

    # その予測を行う
    lin_reg.predict([X_train_scaled[0]])

    # 似たようなフェイクデータを作成
    fake = np.array([-122, 37, 40, 2000, 3000, 500, 3, 3, 6, 4, 0, 0, 1, 0]).reshape(1, -1)

    # トレーニングされたスケーラーを使用して標準化
    std_fake = std_scaler.transform(fake)

    # その観測の予測を試してみる
    lin_reg.predict(std_fake)

    # テストデータセット上で予測を行う
    y_preds = lin_reg.predict(X_test_scaled)
    # 予測を調べる

    # 最初の5つの予測が最初の5つの実際の値とどのように比較されますか？
    true_5 = list(round(y_test[:5], 1))
    pred_5 = [round(x, 1) for x in list(y_preds[:5])]
    print('true values:', true_5)
    print('predicted values:', pred_5)

    # これらの結果をどのように解釈しますか？
    first_5 = ['district0', 'district1', 'district2', 'district3', 'distict4']
    pd.DataFrame(list(zip(first_5, true_5, pred_5)), columns=['district', 'true', 'predicted'])

    # 平均二乗誤差はモデルの平均誤差（$で）を表します
    rmse_ols = np.sqrt(metrics.mean_squared_error(y_test, y_preds))
    rmse_ols = int(rmse_ols)

    # これはコインフリップ（トレーニングセットの平均）と比較してどうなりますか？
    avg_val = round(y_train.mean(), 2)

    # もしそれを予測子として使用した場合、モデルの平均誤差（RMSE）は次のとおりです。
    coinflip_preds = np.full((len(y_test), ), avg_val)
    rmse_coinflip = np.sqrt(metrics.mean_squared_error(y_test, coinflip_preds))
    rmse_coinflip = int(rmse_coinflip)

    # R-squaredはモデルによって説明される従属変数の分散の割合です
    r2_ols = metrics.r2_score(y_test, y_preds)
    r2_ols = round(r2_ols, 2)

    # これはコインフリップ（トレーニングセットの平均）と比較してどうなりますか？
    r2_coinflip = metrics.r2_score(y_test, coinflip_preds)
    r2_coinflip = round(r2_coinflip, 2)

    # OLS線形回帰をベースラインと比較
    evaluation_df = pd.DataFrame([['ベースライン', rmse_coinflip, r2_coinflip],
                                  ['OLS線形回帰', rmse_ols, r2_ols]],
                                 columns=['モデル', 'RMSE', 'R-squared']
                                 )
    evaluation_df.set_index('モデル', inplace=True)

    # Bar chart with plotly: RMSE
    trace = go.Bar(x=list(evaluation_df.index), y=evaluation_df['RMSE'], marker=dict(color=['#E53712', '#1247E5']))
    layout = go.Layout(title='地区別の中央住宅価格：平均二乗平方根誤差',  # グラフのタイトル
                       yaxis=dict(title='モデル'),  # x軸のラベル
                       xaxis=dict(title='RMSE'),  # y軸のラベル
                       )

    rmse_fig = go.Figure(data=[trace], layout=layout)

    # Bar chart with plotly: R-Squared
    trace = go.Bar(x=list(evaluation_df.index), y=evaluation_df['R-squared'], marker=dict(color=['#E53712', '#1247E5']))
    layout = go.Layout(title='地区別の中央住宅価格: R-squared',  # グラフのタイトル
                       yaxis=dict(title='モデル'),  # x軸のラベル
                       xaxis=dict(title='R-Squared'),  # y軸のラベル
                       )

    r2_fig = go.Figure(data=[trace], layout=layout)

    return 0