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
    # to make this notebook's output stable across runs
    np.random.seed(42)
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)

    # read in the dataset
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

    # Create income categories
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    housing["income_cat"].value_counts().sort_index(ascending=True)

    # Create some additional variables
    housing["rooms_per_hhold"] = housing["total_rooms"]/housing["households"]
    housing["pop_per_household"]=housing["population"]/housing["households"]

    housing["rooms_per_hhold"].describe()

    housing["pop_per_household"].describe()

    housing['ocean_proximity'].value_counts().sort_values(ascending=False)

    # Create dummy variables for ocean proximity
    # Note: this is also called "one-hot encoding"
    housing=pd.get_dummies(housing, columns = ['ocean_proximity'], prefix='', prefix_sep='')

    # what are the variables?
    housing.columns

    housing.head()

    # sklearn cannot handle missing data. we're just doing to drop it
    print(len(housing))
    print(housing.isnull().sum())
    housing.dropna(axis=1, inplace=True)
    print(len(housing))

    # First, split your data into features (X) and labels (y).
    y = housing["median_house_value"].copy()
    # We drop one of the 'ocean_proximity' categories so that the coefficients will be interpretable
    X = housing.drop(["median_house_value",'<1H OCEAN'], axis=1)
    # Compare their shapes.
    print(y.shape)
    print(X.shape)

    X.columns

    # Now, split both X and y data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                           test_size=0.2, 
                                           random_state=42)

    # Compare the shapes to confirm this did what you wanted.
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # What are the numeric variables in my dataset?
    X_train.describe().columns

    # "Instantiate" the scaler (create an instance of the sklearn class)
    std_scaler = StandardScaler()

    # 'Fit' the scaler to our X_train data
    std_scaler = std_scaler.fit(X_train.values)

    # Use the scaler to transform the dataset
    X_train_scaled = std_scaler.transform(X_train.values)
    X_train_scaled[0]

    # Use the scaler to transform the dataset
    X_test_scaled = std_scaler.transform(X_test.values)
    X_test_scaled[0]

    # Create a local instance of the sklearn class
    lin_reg = LinearRegression(fit_intercept=True)    

    # Fit your instance to the training dataset
    lin_reg.fit(X_train_scaled, y_train)

    # Check the intercept and coefficients
    print(lin_reg.intercept_)
    print(lin_reg.coef_)

    # 'Attributes' is another name for our list of features (aka predictors, independent variables)
    attributes=X_test.columns
    print(attributes)
    # 'Feature importances' is another name for our coefficients (ie., the impace of each feature on the outcome or DV)
    feature_importances=lin_reg.coef_
    print(feature_importances)

    # obviously, these 2 things will have the same length
    print(len(feature_importances))
    print(len(attributes))

    [int(x) for x in list(feature_importances)]

    # let's take a look at the results
    feature_imp = pd.DataFrame(list(zip(attributes, feature_importances)), columns=['features', 'coeffs'])
    feature_imp=feature_imp.set_index('features')
    feature_imp=feature_imp.sort_values('coeffs')

    # plot that as a bar chart
    feature_imp.plot(kind='bar');

    # with plotly
    data = go.Bar(x=list(feature_imp.index), y=feature_imp['coeffs'])
    coefs = go.Figure([data])    

    X_train.columns

    # Show variables for a single observation:
    print(X_train.iloc[0])

    # Show scaled variables for that single observation:
    # Remember that we dropped the variable "<1H OCEAN"
    print(X_train_scaled[0])

    # Make a prediction on that:
    lin_reg.predict([X_train_scaled[0]])

    # Make up some fake data that's similar
    fake = np.array([-122, 37, 40, 2000, 3000, 500, 3, 3, 6, 4, 0, 0, 1, 0]).reshape(1, -1)

    # Standardize using the trained scaler
    std_fake = std_scaler.transform(fake)

    # Try a prediction for that observation:
    lin_reg.predict(std_fake)

    # Make predictions on the testing dataset
    y_preds = lin_reg.predict(X_test_scaled)
    # Examine your predictions

    # How do the first five predictions compare to the first five actual values?
    true_5=list(round(y_test[:5], 1))
    pred_5=[round(x,1) for x in list(y_preds[:5])]
    print('true values:', true_5)
    print('predicted values:', pred_5)

    # How do we intepret those results?
    first_5=['district0', 'district1', 'district2', 'district3', 'distict4']
    pd.DataFrame(list(zip(first_5, true_5, pred_5)), columns=['district', 'true', 'predicted'])

    # root mean squared error represents the average error (in $) of our model
    rmse_ols = np.sqrt(metrics.mean_squared_error(y_test, y_preds))
    rmse_ols = int(rmse_ols)

    # how does this compare to a coinflip (i.e., the mean of our training set)?
    avg_val = round(y_train.mean(),2)

    # If we used that as our predictor, then the average error (RMSE) of our model would be:
    coinflip_preds=np.full((len(y_test), ), avg_val)
    rmse_coinflip=np.sqrt(metrics.mean_squared_error(y_test, coinflip_preds))
    rmse_coinflip=int(rmse_coinflip)

    # R-squared is the proportion of the variance in the DV that's explained by the model
    r2_ols=metrics.r2_score(y_test, y_preds)
    r2_ols=round(r2_ols, 2)

    # how does this compare to a coinflip (i.e., the mean of our training set)?
    r2_coinflip=metrics.r2_score(y_test, coinflip_preds)
    r2_coinflip=round(r2_coinflip,2)

    # Compare OLS Linear Regression to the Baseline
    evaluation_df = pd.DataFrame([['Baseline',rmse_coinflip, r2_coinflip], 
                                  ['OLS Linear Regression', rmse_ols, r2_ols]], 
                                 columns=['Model','RMSE','R-squared']
                                )
    evaluation_df.set_index('Model', inplace=True)

    # Bar chart with plotly: RMSE
    trace = go.Bar(x=list(evaluation_df.index), y=evaluation_df['RMSE'], marker=dict(color=['#E53712', '#1247E5']))
    layout = go.Layout(title = 'Median House Value by District: Root Mean Squared Error', # Graph title
        yaxis = dict(title = 'Models'), # x-axis label
        xaxis = dict(title = 'RMSE'), # y-axis label  
                      ) 

    fig = go.Figure(data = [trace], layout=layout)

    # Bar chart with plotly: RMSE
    trace = go.Bar(x=list(evaluation_df.index), y=evaluation_df['R-squared'], marker=dict(color=['#E53712', '#1247E5']))
    layout = go.Layout(title = 'Median House Value by District: R-squared', # Graph title
        yaxis = dict(title = 'Models'), # x-axis label
        xaxis = dict(title = 'R-Squared'), # y-axis label  
                      ) 

    fig = go.Figure(data = [trace], layout=layout)

    # Visualize our true vs. predicted values
    plt.figure(figsize=(7,7))
    plt.title('Median House Value by District')
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')
    fig=sns.regplot(x=y_preds, y=y_test)
    plt.show()

    # same thing with plotly
    import plotly.express as px
    fig = px.scatter(y=y_test, x=y_preds, trendline="ols", width=500, height=500)
    fig.update_layout(title = 'Median House Value by District', # Graph title
        yaxis = dict(title = 'True values'), # x-axis label
        xaxis = dict(title = 'Predicted values'), # y-axis label   
    )
    fig.update_traces(line_color='#E53712', line_width=5)
    fig.show()

    
    # Create a local instance of the sklearn class
    ridge_model = linear_model.Ridge(alpha=.5)
    # Fit your instance to the training dataset
    ridge_model.fit(X_train_scaled, y_train)
    # Make predictions on the testing dataset
    y_preds = ridge_model.predict(X_test_scaled)
    # root mean squared error represents the average error (in $) of our model
    ridge_rmse = int(np.sqrt(metrics.mean_squared_error(y_test, y_preds)))
    # R-squared is the proportion of the variance in the DV that's explained by the model
    ridge_r2=round(metrics.r2_score(y_test, y_preds),2)
    print(ridge_rmse, ridge_r2)

    
    # Create a local instance of the sklearn class
    knn_model = KNeighborsRegressor(n_neighbors=8)
    # Fit your instance to the training dataset
    knn_model.fit(X_train_scaled, y_train)
    # Make predictions on the testing dataset
    y_preds = knn_model.predict(X_test_scaled)
    # root mean squared error represents the average error (in $) of our model
    knn_rmse = int(np.sqrt(metrics.mean_squared_error(y_test, y_preds)))
    # R-squared is the proportion of the variance in the DV that's explained by the model
    knn_r2=round(metrics.r2_score(y_test, y_preds),2)
    print(knn_rmse, knn_r2)

    
    # Create a local instance of the sklearn class
    tree_model = DecisionTreeRegressor(max_depth=9)
    # Fit your instance to the training dataset
    tree_model.fit(X_train_scaled, y_train)
    # Make predictions on the testing dataset
    y_preds = tree_model.predict(X_test_scaled)
    # root mean squared error represents the average error (in $) of our model
    tree_rmse = int(np.sqrt(metrics.mean_squared_error(y_test, y_preds)))
    # R-squared is the proportion of the variance in the DV that's explained by the model
    tree_r2=round(metrics.r2_score(y_test, y_preds),2)
    print(tree_rmse, tree_r2)


    # Create a local instance of the sklearn class
    forest_model = RandomForestRegressor(max_depth=10, n_estimators=200)
    # Fit your instance to the training dataset
    forest_model.fit(X_train_scaled, y_train)
    # Make predictions on the testing dataset
    y_preds = forest_model.predict(X_test_scaled)
    # root mean squared error represents the average error (in $) of our model
    forest_rmse = int(np.sqrt(metrics.mean_squared_error(y_test, y_preds)))
    # R-squared is the proportion of the variance in the DV that's explained by the model
    forest_r2=round(metrics.r2_score(y_test, y_preds),2)
    print(forest_rmse, forest_r2)

    # Compare OLS Linear Regression to the Baseline

    evaluation_df2 = pd.DataFrame([['Baseline',rmse_coinflip, r2_coinflip], 
                                  ['OLS Linear Regression', rmse_ols, r2_ols],
                                  ['Ridge Regession', ridge_rmse, ridge_r2],
                                  ['K-Nearest Neighbors Regression', knn_rmse, knn_r2],
                                  ['Decision Tree Regression', tree_rmse, tree_r2],
                                  ['Random Forest Regression', forest_rmse, forest_r2]], 
                                 columns=['Model','RMSE','R-squared']
                                )
    evaluation_df2.set_index('Model', inplace=True) 

    # Bar chart with plotly: RMSE
    trace = go.Bar(x=list(evaluation_df2.index), 
                   y=evaluation_df2['RMSE'], 
                   marker=dict(color=['gray', '#e96060', 'gray', 'gray', 'gray', 'gray']),
    #                marker=dict(color=['#ebc83d','#badf55', '#35b1c9','#b06dad','#e96060', '#1e1d69']),
    #               plot_bgcolor='rgb(10,10,10)'
                  )
    layout = go.Layout(title = 'Model Comparison: Root Mean Squared Error', # Graph title
        yaxis = dict(title = 'Models'), # x-axis label
        xaxis = dict(title = 'RMSE'), # y-axis label  
                      ) 

    rmse_fig = go.Figure(data = [trace], layout=layout)

    # Bar chart with plotly: R-Squared
    trace = go.Bar(x=list(evaluation_df2.index), 
                   y=evaluation_df2['R-squared'], 
                   marker=dict(color=['gray', '#e96060', 'gray', 'gray', 'gray', 'gray']),
    #                marker=dict(color=['#ebc83d','#badf55', '#35b1c9','#b06dad','#e96060', '#1e1d69']),
    #               plot_bgcolor='rgb(10,10,10)'
                  )
    layout = go.Layout(title = 'Model Comparison: R-Squared', # Graph title
        yaxis = dict(title = 'Models'), # x-axis label
        xaxis = dict(title = 'R-Squared'), # y-axis label  
                      ) 

    r2_fig = go.Figure(data = [trace], layout=layout)

    return 0