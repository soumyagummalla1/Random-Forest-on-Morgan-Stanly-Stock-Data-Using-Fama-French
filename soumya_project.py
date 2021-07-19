# -*- coding: utf-8 -*-
"""Soumya_Project.ipynb

# CAPM, Fama French(3 Factor and 5 Factor)

In this notebook we will be implementing 3 models to determine required rate of return of an asset, to make decisions about adding assets to a well-diversified portfolio.

CAPM: - Capital Asset Pricing Model (CAPM) is a model that describes the relationship between the expected return and risk of investing in a security. It shows that the expected return on a security is equal to the risk-free return plus a risk premium, which is based on the beta of that security.

Fama French(3-Factor): - The Fama-French Three-factor Model is an extension of the Capital Asset Pricing Model (CAPM). The Fama-French model aims to describe stock returns through three factors: (1) market risk, (2) the outperformance of small-cap companies relative to large-cap companies, and (3) the outperformance of high book-to-market value companies versus low book-to-market value companies. The rationale behind the model is that high value and small-cap companies tend to regularly outperform the overall market.

### Importing Essential Libraries
"""

import pandas as pd
import numpy as np 

import statsmodels.api as sm 
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

"""### Defining Python functions

We have two functions below:

1. price2ret: This converts prices to arithmetic or log returns.
2. assetPriceReg: By giving a dataframe of stock with a column named Returns, the function extracts the risk factor returns from web and runs a CAPM, FF3, and FF5 regression
"""

def price2ret(prices,retType='simple'):
    if retType == 'simple':
        ret = (prices/prices.shift(1))-1
    else:
        ret = np.log(prices/prices.shift(1))
    return ret

def assetPriceReg(df_stk):
    import pandas_datareader.data as web 
    
    # Reading in factor data
    df_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench')[0]
    df_factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)
    df_factors['MKT'] = df_factors['MKT']/100
    df_factors['SMB'] = df_factors['SMB']/100
    df_factors['HML'] = df_factors['HML']/100
    df_factors['RMW'] = df_factors['RMW']/100
    df_factors['CMA'] = df_factors['CMA']/100
    
    df_stock_factor = pd.merge(df_stk,df_factors,left_index=True,right_index=True) 
    df_stock_factor['XsRet'] = df_stock_factor['Returns'] - df_stock_factor['RF'] 

    # Running CAPM, FF3, and FF5 models.
    CAPM = smf.ols(formula = 'XsRet ~ MKT', data=df_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    FF3 = smf.ols( formula = 'XsRet ~ MKT + SMB + HML', data=df_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    FF5 = smf.ols( formula = 'XsRet ~ MKT + SMB + HML + RMW + CMA', data=df_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})

    CAPMtstat = CAPM.tvalues
    FF3tstat = FF3.tvalues
    FF5tstat = FF5.tvalues

    CAPMcoeff = CAPM.params
    FF3coeff = FF3.params
    FF5coeff = FF5.params

    # DataFrame with coefficients and t-stats
    results_df = pd.DataFrame({'CAPMcoeff':CAPMcoeff,'CAPMtstat':CAPMtstat,
                               'FF3coeff':FF3coeff, 'FF3tstat':FF3tstat,
                               'FF5coeff':FF5coeff, 'FF5tstat':FF5tstat},
    index = ['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])


    dfoutput = summary_col([CAPM,FF3, FF5],stars=True,float_format='%0.4f',
                  model_names=['CAPM','FF3','FF5'],
                  info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                             'Adjusted R2':lambda x: "{:.4f}".format(x.rsquared_adj)}, 
                             regressor_order = ['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])

    print(dfoutput)
    
    return results_df

"""Reading in stock data, removing volume information, plotting prices and Calculating simple returns from stock prices"""

# Commented out IPython magic to ensure Python compatibility.
from pathlib import Path
import sys
import os
import pandas_datareader.data as web
from pandas import Series, DataFrame

# %matplotlib inline

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

import datetime

start = datetime.datetime(2009, 1, 1)
end = datetime.datetime(2016, 1, 11)

df_stk = web.DataReader("BMY", 'yahoo', start, end)
df_stk.shape

Manipulated data is shown below:

df_stk.head()

applying the price to return function on 'Adj Close' and dropped nas and the information is shown below

df_stk['Returns'] = price2ret(df_stk[['Adj Close']])
df_stk = df_stk.dropna()
df_stk.tail()

we plot returns against date columns and observe the pattern

df_stk['Returns'].plot()

Similarly a distribution plot has been plotted for returns which shows that the dist is normal

df_stk['Returns'].hist(bins=20)

"""Running risk factor regressions"""

df_regOutput = assetPriceReg(df_stk)

df_regOutput

"""# Random Forest"""

Data has been loaded and required columns are scaled dividing by 100.The column MKT-RF has been renamed to MKT

# Reading in factor data
    df_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench')[0]
    df_factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)
    df_factors['MKT'] = df_factors['MKT']/100
    df_factors['SMB'] = df_factors['SMB']/100
    df_factors['HML'] = df_factors['HML']/100
    df_factors['RMW'] = df_factors['RMW']/100
    df_factors['CMA'] = df_factors['CMA']/100

df_factors

Now we merge two datasets namely df_stk, df_factors using merge operation where the kind is inner merge.

merge=pd.merge(df_stk, df_factors, how='inner', left_index=True, right_index=True)

We drop the returns column as follows

merge = merge.drop(['Returns'], axis=1)

And perform renaming column 'MKT' -> 'MKt-RF'

merge = merge.rename({'MKT': 'MKt-RF'}, axis = 1)

The data is displayed as below:

merge

We now import the required libraries as below

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pprint import pprint
from sklearn.metrics import mean_squared_error

The function getData reads in data with from and to dates and perform aggregations such as average, no of days, diff_open, diff_close etc

def getData(strt_date = "2015-11-16", end_date = "2016-01-11"):
    
    main = pd.read_csv('merge.csv')
    
    FF["Date"] = pd.to_datetime(FF["Date"])
    
    df = FF[(FF["Date"] >= strt_date) & (FF["Date"] <= end_date)]
    
    df = df.reset_index()
    df = df.drop(["index"], axis =1)
    
    df["Average"] = (FF["Open"] + 2*FF["High"] + FF["Close"])/4
    
    df["Days"] = range(1,df.shape[0]+1)
    
    df["diff_Open"] = (df.Open - df.Open.shift(periods=1))
    
    df["diff_Close"] = (df.Close - df.Close.shift(periods=1))
    
    df["diff_RMW"] = (df.RMW - df.RMW.shift(periods=1))
    
    df["diff_SMB"] = (df.SMB - df.SMB.shift(periods=1))
    
    df["diff_MktRF"] = (df["MKt-RF"] - df["MKt-RF"].shift(periods=1))
    
    df["diff_HML"] = (df.HML - df.HML.shift(periods=1))
    
    df["diff_CMA"] = (df.CMA - df.CMA.shift(periods=1))
    
    df["diff_Average"] = (df.Average - df.Average.shift(periods=1))
    df = df.dropna().reset_index(drop=True)
    
    df["label"] = df.delta_Open
    df = df.drop("delta_Open", axis=1)
    
    return df

df = getData()

Below are the new columns introduced

df.columns

Now we modify the data by dropping "label" remove "na" and reset the index for df_x and manipulations such as dropna(), reser_index on df_y i.e taget column and merge both and stored in as modified_df

df_x = df.drop("label", axis =1).shift(periods=1).dropna().reset_index(drop=True)
df_y = df[["label"]].shift(periods=-1).dropna().reset_index(drop=True)
modified_df = pd.concat([df_x,df_y], axis =1)

modified_df

Below we have given required train and test , start and end dates respectively now we chop the required data according to start and end date and take in required columns such as open, volume, rf, diff_close,diff_RMW etc

train_start_date = "2015-11-17"
train_end_date =  "2015-12-28"
test_start_date = "2015-12-29"
test_end_date = "2016-01-08"
df_tr = df[(df["Date"] >= train_start_date) & (df["Date"] <= train_end_date)]
df_tst = df[(df["Date"] >= test_start_date) & (df["Date"] <= test_end_date)]
df_train = df_tr[["Open","Volume","RF","diff_Close","diff_RMW","diff_SMB","diff_MktRF","diff_HML","diff_CMA","diff_Average","label"]]
df_test = df_tst[["Open","Volume","RF","diff_Close","diff_RMW","diff_SMB","diff_MktRF","diff_HML","diff_CMA","diff_Average","label"]]

df_train.head()

df_test.head()

Below is a function which checks the purity such as unique values returns true if the len(unique_class) is 1 else false

def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    else:
        return False

The below function creates leaf node which is the mean if the column and return the value

def create_leaf(data):
    label_column = data[:, -1]
    leaf = np.mean(label_column)
    return leaf

The get_potential_splits is used to split a random space and return the required potential split of data

def get_potential_splits(data, random_subspace):
    
    potential_splits = {}
    _, n_columns = data.shape
    
    column_indices = list(range(n_columns-1))       
    
    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_subspace)
    
    for column_index in column_indices:          
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
        
    return potential_splits

split data functions splits the columns as follows if the type_of_feature is "continuous" 
then we have data_below consisting split_column_values <= split_value and data_above as split_column_values> split_value
if the feature is categorical our split is as follows data_below
has split_col_value == split_value else != split_value and required data is returned from the function

def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    
    # feature is categorical   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above

we need measure our model performance in this case we use calculate_mse in order to calculate mean square error on data there are two conditions
if actual_values are zero then there is no comparision among predicted and actual hence mse = 0 
else we formulate mse as mean_square of difference of actual and preducted

def calculate_mse(data):
    
    actual_values = data[:, -1]
    if len(actual_values) == 0:   # empty data
        mse = 0
    else:
        prediction = np.mean(actual_values)
        mse = np.mean((actual_values - prediction) **2)
    
    return mse

Now we need to caluclate the overall_performance of both above and below data which is as follows
we use weighted sum of data and metrics and return overall_value

def calculate_overall_metric(data_below, data_above, metric_function):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_metric =  (p_data_below * metric_function(data_below) 
                     + p_data_above * metric_function(data_above))
    
    return overall_metric

In order to determine the best split we use the below function which describes the best split by considering the amount of mse or metrics returns due to the specific split . This in turn returns the best split whihc has least error

def determine_best_split(data, potential_splits):
    
    first_iteration = True
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_metric = calculate_overall_metric(data_below, data_above, metric_function=calculate_mse)

            if first_iteration or current_overall_metric <= best_overall_metric:
                first_iteration = False
                
                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

This function  checks whether a feature (except label) is categorical or continuous

def determine_type_of_feature(df):
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types

Bootstrapping is used for randomization where we provide bootstrap length and the indices are assigned randomly 
and requied bootstrapped data will be returned as df_bootstrapped

def bootstrapping(train_df, n_bootstrap):
    
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    
    return df_bootstrapped

Decision_tree method takes in counter, datatset, min_samples, max_samples and random_subspace if the counter whihc acts like flag is set to zero whihc is our first iteration the data is taken in and from counter > 0 the df is assigned to data variable , 
the required checks such as purity, best_split, etc are used and sub_tree is returned

def decision_tree_algorithm(df, counter=0, min_samples=5, max_depth=5, random_subspace=None):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        leaf = create_leaf(data)
        return leaf

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data, random_subspace)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            leaf = create_leaf(data)
            return leaf
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth, random_subspace)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth, random_subspace)
        
        # If yes_answer = no_answer
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree

Now we apply bootstrapping on random tree and return random_forest on dataset

def random_forest_algorithm(df_train, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(df_train, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)
    
    return forest

we fit our model with an example

def predict_example(example, tree):
    record = list()
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)

decision tree predivtions are performed on test data

def decision_tree_predictions(df_test, tree):
    predictions = df_test.apply(predict_example, args=(tree,), axis=1)
    return predictions

random_forest_predictions are performed on test data as follows

def random_forest_predictions(df_test, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(df_test, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    
    return random_forest_predictions

Performance metrics such as r_squared, rmse are calculated below:

def calculate_r_squared(predictions, labels):    
    mean = labels.mean()
    ss_res = sum((labels - predictions) ** 2)
    ss_tot = sum((labels - mean) ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    return r_squared

def calculate_rmse(predictions, labels):
    mse = mean_squared_error(labels,predictions)
    rmse = np.sqrt(mse)
    return rmse

A plot is built between actuals and predictions as follows

def create_plot(predictions, labels, title):    
    plot_df = pd.DataFrame({"actual": labels, "predictions": predictions})
    plot_df.plot(figsize=(18, 5), title=title)
    
    return

grid_search = {"max_depth": [], "trees": [], "rmse_test": [], "r_squared_test": [], "bootstrap": [] }
for max_depth in range(1, 7):
    for trees in range(1,10):
        for bootstrap in range(25,225,25):
            forest = random_forest_algorithm(df_train, n_trees=trees, n_bootstrap= bootstrap, n_features=3, dt_max_depth=max_depth)
            delta_Open_pred = random_forest_predictions(df_test, forest)
            df_pred = pd.DataFrame(list(zip(delta_Open_pred,df_test.label.values, df_test.Open.values)), columns = ["predictions","Actual","Open_Act"])
            df_pred["Open_pred"] = df_pred.predictions + df_test.Open.shift(1).reset_index(drop=True)
            df_pred = df_pred.dropna().reset_index(drop=True)

            rmse_test = calculate_rmse(list(df_pred.Open_pred.values), list(df_pred.Open_Act.values))
            r_squared_test = calculate_r_squared(df_pred.Open_pred, df_pred.Open_Act)

            grid_search["max_depth"].append(max_depth)
            grid_search["trees"].append(trees)
            grid_search["bootstrap"].append(bootstrap)
            grid_search["rmse_test"].append(rmse_test)
            grid_search["r_squared_test"].append(r_squared_test)
        
    print(f"Progress: Iteration {max_depth}/6")
        
grid_search = pd.DataFrame(grid_search)
grid_search.sort_values("r_squared_test", ascending=False).head()

forest = random_forest_algorithm(df_train, n_trees=2, n_bootstrap= 75, n_features=3, dt_max_depth=5)
delta_Open_pred = random_forest_predictions(df_test, forest)
df_pred = pd.DataFrame(list(zip(delta_Open_pred,df_test.label.values, df_test.Open.values)), columns = ["predictions","Actual","Open_Act"])
df_pred["Open_pred"] = df_pred.predictions + df_test.Open.shift(1).reset_index(drop=True)
df_pred = df_pred.dropna().reset_index(drop=True)

rmse_test = calculate_rmse(list(df_pred.Open_pred.values), list(df_pred.Open_Act.values))
r_squared_test = calculate_r_squared(df_pred.Open_pred, df_pred.Open_Act)
    
print("RMSE value of best possible solution: ",rmse_test)
print("R2 value of best possible solution: ",r_squared_test)

create_plot(list(df_pred.Open_pred.values), list(df_pred.Open_Act.values), title="Test Data")

