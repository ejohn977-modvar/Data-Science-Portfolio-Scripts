import pandas as pd
import numpy as np
np.random.seed(101)
from scipy.interpolate import interp1d
import itertools
import time
import sys
import os
from datetime import date
import calendar

import psycopg2
import psycopg2.extras
from dateutil import parser
import csv
from unidecode import unidecode
import configparser
ini_file = str(sys.argv[1])
config = configparser.ConfigParser()
config.sections()
config.read(ini_file)

##########################################################
# Date and Time printout at start of running script.
from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
#print("now =", now)
# dd/mm/YY H:M:S
t0 = time.time()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)


#########################################################

def get_dataframe(sql_string: str):
    db_dbname = str(config['Database']['db_dbname'])
    db_user = str(config['Database']['db_user'])
    db_password = str(config['Database']['db_password'])
    db_host = str(config['Database']['db_host'])
    try:
        conn=psycopg2.connect("dbname=" + (db_dbname) + "user=" + (db_user)+"password=" + (db_password)+ "host=" + (db_host))
        conn.autocommit = False
        print ("connection opened to postgres")

    except psycopg2.DatabaseError as exception:
        print("Postgres Error: " + str(exception))

    try:
        sql_query =  sql_string
        df_try = pd.read_sql_query(sql_query, conn)
    except:
        print("Something went wrong with the query.")
    return df_try
def pre_process(group):
    group = pd.DataFrame(group.resample('D', kind = 'timestamp').asfreq())
    group['adjclose_val'] = group['adjclose_val'].fillna(-1)
    return group
def impute_with_noise(df):
    temp = df['adjclose_val'].to_list()
    Flag = True
    x = 0
    ct = 0
    next_non_nan_index = 0
    while(Flag):
        if temp[x] == -1.0:
            try:
                next_non_nan_index = next(i for i, j in enumerate(temp[x:]) if j != -1) + x
            except:
                Flag = False
                #print(x)
            if Flag:
                index_diff = (next_non_nan_index - x) + 1
                impute_vals = np.linspace(temp[x-1], \
                    temp[next_non_nan_index], num = index_diff, endpoint = False)
                impute_vals = impute_vals + \
                    np.random.normal(0,.05*(abs(temp[x-1] - temp[next_non_nan_index])), len(impute_vals))
                impute_vals = impute_vals.tolist()
                temp[x-1:next_non_nan_index] = impute_vals
                x = next_non_nan_index
                #print(x)
        else:
            x = x+1
            #print(x)
        if x == len(temp)-1:
            Flag = False
        if ct == len(temp) + 10:
            Flag = False
        ct += 1
        #print(ct)
    s = pd.Series(temp,index = df.index)
    return s
# calculate price volatility array given company
def calculate_price_volatility(stock_num_days,index_num_days,m_days_ahead, price_array,is_index: bool):
    # make price volatility array
    volatility_array = []
    offset = stock_num_days
    temp = price_array

    if is_index:
        offset = index_num_days

    for i in range(max(stock_num_days,index_num_days)-offset, len(price_array) - offset - m_days_ahead, 1):
        try:
            percent_change = 100 * temp[i:i+stock_num_days].pct_change()
            percent_change.dropna(inplace = True)
            volatility_array.append(percent_change.mean())
        except Exception:
            volatility_array.append(-1000000)

    return volatility_array
# calculate momentum array
def calculate_price_momentum(stock_num_days,index_num_days,m_days_ahead,price_array,is_index: bool):
    # now calculate momentum
    momentum_array = []
    direction_array = []
    offset = stock_num_days
    if is_index:
        offset = index_num_days
    for i in range(max(stock_num_days,index_num_days)-offset, len(price_array) - offset - m_days_ahead, 1):
        temp = price_array[i:i+offset]
        direction_array = (temp - temp.shift(1)).dropna()
        direction_array = np.where(direction_array > 0, 1, -1)
        try:
            momentum_array.append(np.mean(direction_array))
        except Exception:
            momentum_array.append(-1000000)
    #print("mom length: ", len(momentum_array))
    return momentum_array
def calculate_change_m_days_ahead(stock_num_days, index_num_days, m_days_ahead, price_array, is_index: bool):
    """ Look into using continuous label and doing regression instead. """
    labels = []
    offset = stock_num_days
    temp = price_array
    if is_index:
        offset = index_num_days
    for i in range(max(stock_num_days,index_num_days)-offset, len(temp)- offset - m_days_ahead, 1):
        try:
            if temp[i+offset+m_days_ahead] > temp[i+offset]:
                labels.append(1)
            else:
                labels.append(0)
        except:
            pass

    return np.array(labels, dtype = 'int')
    
def make_train_features(stock_num_days,index_num_days,m_days_ahead,df_stock):
    #overlap = df_stock.index.intersection(df_index.index)
    #df_stock = df_stock.loc[overlap[0]:overlap[-1]].copy()
    #df_index = df_index.loc[overlap[0]:overlap[-1]].copy()
    stock_volatility_array = list(calculate_price_volatility(stock_num_days,index_num_days,m_days_ahead,\
        df_stock,is_index = False))
    stock_momentum_array = list(calculate_price_momentum(stock_num_days,index_num_days,m_days_ahead,\
        df_stock,is_index = False))

    features_transpose = [stock_volatility_array,stock_momentum_array]
    features_transpose = np.array(features_transpose, dtype = 'float64')
    return np.transpose(features_transpose)


def ready_to_model_symbol_level(stock_df, stock_num_days, index_num_days, m_days_ahead):
    stock_df = pre_process(stock_df)
    stock_df = impute_with_noise(stock_df)
    features = \
        make_train_features(stock_num_days, index_num_days, m_days_ahead, stock_df)
    direction_m_days_ahead = \
        calculate_change_m_days_ahead(stock_num_days,index_num_days,m_days_ahead,stock_df,is_index = False)

    return features, direction_m_days_ahead
##############################################################################################################
"""EJ 01/12/2021: This is where the main script and pre-processing work starts. """

stock_num_days = 90
index_num_days = 90
m_days_ahead = 5
###########################################################
"""EJ 01/12/2021: The section below is imputing price points for the S&P 500 index and 
then calcualting the volatility and momentum array for it. """

""" EJ 01/17/2021: put paths into ini_file"""
df_index = \
pd.read_csv("C:\\Users\\Rachel\\Documents\\GitRepo\\s&p500_20year_daily_price_data\\sp500_index.csv")
df_index['date'] = pd.to_datetime(df_index['Date'])
df_index['adjclose_val'] = df_index['Adj Close']
df_index = df_index[['date','adjclose_val']].copy()
df_index.set_index('date', inplace = True)
df_index = df_index['2013-01-01':'2020-12-21'].copy()
pre_impute_index_length = len(df_index.index)
#pre_impute_test_length = len(df_index['2020-01-01':].index)


df_index = pre_process(df_index)
df_index = impute_with_noise(df_index)

#df_index = pd.DataFrame(df_index['2014-01-01':'2019-12-31'])
#df_index.rename(columns = {0:'adjclose_val'}, inplace = True)
index_len = len(df_index.index)
""" df_index_test = pd.DataFrame(df_index['2020-01-01':])
df_index_test.rename(columns = {0:'adjclose_val'}, inplace = True)
test_ind_len = len(df_index_test.index) """


a= max(stock_num_days,index_num_days)
df_index1 = pd.DataFrame(df_index.iloc[a+1:1 + index_len  -m_days_ahead])
df_index1.columns = ['adjclose_val']

""" df_index_test1 = pd.DataFrame(df_index_test.iloc[a+1:1+test_ind_len-m_days_ahead,:])
df_index_test1.columns = ['adjclose_val'] """


df_index1['price_vol'] = np.array(calculate_price_volatility(stock_num_days,index_num_days,m_days_ahead,\
df_index,is_index = True))
#print(df_index_train1.columns)
#print(df_index_train1.head(10))
df_index1['price_mom'] = np.array(calculate_price_momentum(stock_num_days,index_num_days,m_days_ahead,\
df_index,is_index = True))
#print(df_index_train1.columns)
#print(df_index_train1.head(10))


df_index1.drop(columns = 'adjclose_val', inplace = True)
index_np_values = df_index1.to_numpy()
#df_index_test1.drop(columns = 'adjclose_val', inplace = True)
#print(df_index_test1.head(10))
"""EJ 01/12/2021: This is where the preprocessing and feature extraction ends for the index dataframe. """
#############################################################

#############################################################
"""EJ 01/12/2021: This next chunk of code does the pre-processing and feature extraction for
the companies of interest in the model and then combines company and index features to feed
into the radial basis function SVC. """
t1 = time.time()
print("Total df_index creation time: ", t1 - t0)

# Set the SQL call to feed in to the get_dataframe function below.
sql_call = """select symbol, date, adjclose_val from  stockmarket_dailyvalues 
        where date between '2013-01-01' and '2020-12-21';"""

df_all = get_dataframe(sql_call)
"""df_all = pd.DataFrame(df_all.loc[df_all['date'] >= '2014-01-01'])
df_train = pd.DataFrame(df_all.loc[df_all['date'] < '2020-01-01'])
# test dataframe
df_test = pd.DataFrame(df_all.loc[df_all['date'] >= '2020-01-01'])
#print(df_all_train.tail(10)) """

df_grouped_by_symbol = df_all.groupby('symbol', as_index=True)

x_train = []
y_train = []
x_test = []
y_test = []


##############################################################
""" EJ 01/17/2021: Change to only grabbing symbols in list, then loop over symbol list and pull view of dataframe
at that symbol and run methods from there. Thanks Kolin! """ 
# Randomly sampling from stock price list
import random
groups = [group for name, group in df_grouped_by_symbol]
groups = random.sample(groups, 20)
##############################################################

company_ct = 0
for company in groups:
    #print(company['adjclose_val'].count(), pre_impute_index_length)
    if company['adjclose_val'].count() == pre_impute_index_length:
        company_ct += 1
        company.set_index('date', inplace = True)
        company.drop(columns = 'symbol', inplace = True)
        company_features, company_feature_labels = ready_to_model_symbol_level(company, \
            stock_num_days, index_num_days, m_days_ahead)
        company_features = np.append(company_features, index_np_values, axis = 1)

        company_train_features = company_features[0:int(0.8*len(company_features))]
        company_test_features = company_features[int(0.8*len(company_features)):]
        company_train_feature_labels = company_feature_labels[0:int(0.8*len(company_features))]
        company_test_feature_labels = company_feature_labels[int(0.8*len(company_features)):]

        for i in range(0,len(company_train_features)):
            x_train.append(company_train_features[i])
            y_train.append(company_train_feature_labels[i])

        for i in range(0,len(company_test_features)):
            x_test.append(company_test_features[i])
            y_test.append(company_test_feature_labels[i])

""" EJ 01/17/2021: Justify this step more..."""
#TODO

x_train = np.ascontiguousarray(x_train, dtype=np.float32)
y_train = np.ascontiguousarray(y_train, dtype = np.int)
x_test = np.ascontiguousarray(x_test, dtype=np.float32)
y_test = np.ascontiguousarray(y_test, dtype = np.int)

# Simple check of proportion of train and test label arrays that have class label 1.
train_prop = 0
for y in y_train:
    train_prop = train_prop + y
if len(y_train) > 0:
        train_prop = train_prop/len(y_train)

test_prop = 0
for y in y_test:
    test_prop = test_prop + y
if len(y_test) > 0:
        test_prop = test_prop/len(y_test)

t2 = time.time()
print("Total feature generation time: ", t2 - t1)
print("Number of training examples generated: ", len(x_train))
print("Number of test cases: ", len(x_test))
print("Proportion of train labels 1: ", train_prop)
print("Proportion of test labels 1: ", test_prop)
########################################################


##############################################################################################################
"""EJ 01/12/2021: Below will start the actual training of the SVC and its predictions and evaluations. """
#TODO Wrap rfflearn into function for handwaving

import rfflearn.cpu as rfflearn

gpc = rfflearn.RFFGPC().fit(x_train, y_train)
y_pred = gpc.predict(x_test)
print("Mean accuracy score: ", gpc.score(x_train, y_train))
#print(y_pred)
t3 = time.time()


""" svc = rfflearn.RFFSVC().fit(x_train, y_train)
print(svc.score(x_test, y_test) )
y_pred = svc.predict(x_test) """

from sklearn.metrics import confusion_matrix
#print(y_test[0:100])

# Some basic metrics
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ", confusion_mat)
from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, y_pred, normalize=True)
print("scikit-learn accuracy_score: ", accuracy_score)
from sklearn.metrics import recall_score
recall_score = recall_score(y_test, y_pred, average='macro')
print("scikit-learn recall_score: ", recall_score)

# Basic script info
print("Total rfflearn run time: ", t3 - t2)
#print(x_train[0:10])
print("Total number of stocks used: ", company_ct)
t4 = time.time()
print("Total run time: ", t4 - t0)

##############################################################################################################
