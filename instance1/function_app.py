''' 
インスタンスの起動数が応答速度に与える影響を調査。
コードは評価で用いるコードを利用し、データの転送サイズが0になるようにして実装。
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
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
import seaborn as sns
import plotly.express as px


app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)
### クライアント関数 ###
@app.route(route="orchestrators/client_function")
@app.durable_client_input(client_name="client")
async def client_function(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    act = int(req.params.get('act') or req.get_json().get('act'))
    split = int(req.params.get('split') or req.get_json().get('split'))
    instance_id = await client.start_new("orchestrator", None, {"act": act, "split": split})
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)
    return client.create_check_status_response(req, instance_id)

### オーケストレーター関数 ###
@app.orchestration_trigger(context_name="context")
def orchestrator(context: df.DurableOrchestrationContext) -> str:
    parameter = context.get_input()
    act = int(parameter.get("act"))
    split = int(parameter.get("split")) # 分割数取得
    
    if act == 0:
        # 複数のインスタンスを経由
        for i in range(split):
            result = yield context.call_activity("a_code", '')    
    elif act == 1:
        # 1つのインスタンスで何度も実行
        result = yield context.call_activity("loop", split)
    elif act == 2:
        result = yield context.call_activity("test", '')
    
    return 0

#################################################
### 探索的データ分析 ###
@app.blob_input(arg_name="inputblob", path="dataset/housing.csv", connection="BlobStorageConnection")
@app.activity_trigger(input_name="blank")
def a_code(blank: str, inputblob: func.InputStream):

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

    evaluation_df2 = pd.DataFrame([['ベースライン',rmse_coinflip, r2_coinflip],
                                      ['OLS線形回帰', rmse_ols, r2_ols],
                                      ['リッジ回帰', ridge_rmse, ridge_r2],
                                      ['K近傍法回帰', knn_rmse, knn_r2],
                                      ['決定木回帰', tree_rmse, tree_r2]],
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

    return 0

@app.blob_input(arg_name="inputblob", path="dataset/housing.csv", connection="BlobStorageConnection")
@app.activity_trigger(input_name="split")
def loop(split: int, inputblob: func.InputStream) -> dict:
    origin = pd.read_csv(inputblob)
    for i in range(split):
        # 出力を再現可能にするためにシードを設定
        np.random.seed(42)
        mpl.rc('axes', labelsize=14)
        mpl.rc('xtick', labelsize=12)
        mpl.rc('ytick', labelsize=12)

        # データセットの読み込み
        housing = origin.copy()
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

        evaluation_df2 = pd.DataFrame([['ベースライン',rmse_coinflip, r2_coinflip],
                                          ['OLS線形回帰', rmse_ols, r2_ols],
                                          ['リッジ回帰', ridge_rmse, ridge_r2],
                                          ['K近傍法回帰', knn_rmse, knn_r2],
                                          ['決定木回帰', tree_rmse, tree_r2]],
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

    return 0

