import pandas as pd
import numpy as np
import time
import sys
import os
from datetime import date
import calendar
t0 = time.time()

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
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
#########################################################

# Reading in table with S&P 500 company tickers.
""" table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]
ticker_list = df[0].tolist()
del ticker_list[0] """

# EJ 12/29/2020: Weird string storage issue from downloading and using the table above. 
# The next 5 lines fix it.
#print(ticker_list.index('BRK.B'))
#print(ticker_list.index('BF.B'))

""" ticker_list[65] = 'BRK-B'
ticker_list[78] = 'BF-B'
ticker_list.sort()
#print(ticker_list)

# Shortened ticker list created to experiment and debug before full runs.
short_list = ticker_list """

# A bunch of lists some used and some not potentially.
#train_features = []
#test_features = []
train_labels = []
test_labels = []

train_weeks = []
test_weeks = []
df_all = pd.DataFrame()

# EJ 12/29/2020: Two lists below added to find which symbol level dataframe was throwing an error. 
# Maybe no training timeframe data for the culprit company.
grouped_train_added = []
grouped_test_added = []



""" EJ 12/29/2020: Whatever is in the block outlined in pound signs below will 
read in data into one large dataframe. By whatever I just mean 
whether it is the call to the database or manually pulling in all of the csv files."""
####################################################################################################

# EJ 12/29/2020: Block strung out below is sql query to pull from the database.
# I have not gotten a chance to test the database query code bcs of ip address issue.
db_dbname = str(config['Database']['db_dbname'])
db_user = str(config['Database']['db_user'])
db_password = str(config['Database']['db_password'])
db_host = str(config['Database']['db_host'])

try:
    conn=psycopg2.connect("dbname=" + (db_dbname) + 
                            "user=" + (db_user)
                            +"password=" + (db_password)
                            + "host=" + (db_host))
    conn.autocommit = False
    cur = conn.cursor()
    print ("connection opened to postgres")

except psycopg2.DatabaseError as exception:
    print("Postgres Error: " + str(exception))

try:
    sql_query =  """select symbol, date, adjclose_val from  stockmarket_dailyvalues"""
    
    df_all = pd.read_sql_query(sql_query, conn)

except:
    print("Something went wrong with the query.")


# Code in for loop below pulls in all data from stock price csv files and sticks into one large dataframe.
""" for ticker in short_list:
    df_temp = pd.read_csv("C:\\Users\\Rachel\\Documents\\GitRepo\\s&p500_20year_daily_price_data\\"+ticker+".csv")
    
    \""" Create loop, read in CSV files and turn them into dataframes. Then create new dataframe that holds
    date-time index and adjusted closing price. Feed into the grouper to groupby trading weeks. Keep only the
    values from that week as a vector. Store in weeks. Repeat for new ticker history.

    Once done, go through and add labels to label array. After all is completed, split into test and train
    and drop feature weeks (and their corresponding labels) that do not have 5 consecutive trading days. 
    Once done with this we feed into SVM with default kernel = RBF, or some other classifier. \"""

    df_temp['symbol'] = ticker

    df_all = df_all.append(df_temp) """
####################################################################################################


"""EJ 12/28/2020: Code block below corrects misnamed date column and splits by date to create train
and test dataframes. The train and test dataframes are grouped by symbol. Then we take each symbol dataframe
and group by weekly chunks of time using the pd.Grouper(freq = '1w'). We store a week's worth of 
adjusted closing price data as an array and store each array in the array train_weeks 
and test_weeks respectively.  """
##################################################################################################
""" df_all['date'] = pd.to_datetime(df_all['Unnamed: 0']) 
df_all = df_all.drop(columns = 'Unnamed: 0') """
# training dataframe
df_all_train = pd.DataFrame(df_all.loc[df_all['date'] < '2016-01-01'])
# test dataframe
df_all_test = pd.DataFrame(df_all.loc[df_all['date'] >= '2016-01-01'])
#print(df_all_train.tail(10))

# grouping at symbol level for training data set.
df_all_train_grouped = df_all_train.groupby(['symbol'])
groups = [group for name, group in df_all_train_grouped]
''' tests = []
for group in groups[0:1]:
    group = group.set_index('date')
    tests = tests + [((g['adjclose_val'] - g['adjclose_val'][0])/g['adjclose_val'][0]).values \
        for n,g in group.groupby(pd.Grouper(freq = '1w'))]
print(tests)'''

# splitting each symbol dataframe into weekly chunks and keeping only adjusted close price for the week
#  as an array.
for group in groups:
    try:
        group = group.set_index('date')
        train_weeks = train_weeks + [((g['adjclose_val'] - g['adjclose_val'][0])/g['adjclose_val'][0]).values \
            for n,g in group.groupby(pd.Grouper(freq = '1w'))]
    except:
        print(group.columns, group.index, group)

# grouping at symbol level for test data set.
df_all_test_grouped = df_all_test.groupby(['symbol'])
groups = [group for name, group in df_all_test_grouped]

# splitting each symbol dataframe into weekly chunks and keeping only adjusted close price for the week
#  as an array.
for group in groups:
    try:
        group = group.set_index('date')
        test_weeks = test_weeks + [((g['adjclose_val'] - g['adjclose_val'][0])/g['adjclose_val'][0]).values \
            for n,g in group.groupby(pd.Grouper(freq = '1w'))]
    except:
        print(group.columns, group.index, group)

####################################################################################################


"""The following block of code creates the class labels for the weekly adjusted closing price features.
The class labels are 0 or 1 and are derived from the maximum percent increase of the adjusted closing price
over the course of the week following a correspoding weekly training feature.

As a first pass, the classes are 0 for price did not rise above 2% from first day adjusted close and
1 for the adjusted closing price having risen above 2%.

Example: for a given 5-d vector that contains adjusted close price for each day of the week,
its class label is 0 if the (possibly not 5-d vector depending on how many trading days there were that week)
following week's adjusted maximum adjusted close price does not rise 2% above the adjusted close price
from the first day of the class label generating week and 1 if it does."""
# Generating class labels for training weeks
for x in range(0,len(train_weeks)):
    if (x > 0) and (max(train_weeks[x]) > 0.02):
        train_labels.append(1)
    elif (x > 0):
        train_labels.append(0)

# This if else block creates class label for the last week of data in the training array.
if max(test_weeks[0]) > 0.02:
    train_labels.append(1)
else:
    train_labels.append(0)

# This for loop creates the class labels for the test set.
for x in range(0,len(test_weeks)):
    if (x > 0) and (max(test_weeks[x]) > 0.02):
        test_labels.append(1)
    elif (x > 0):
        test_labels.append(0)

# We must drop the last week from the test set as there is no ability to create
# a class label for it.
del test_weeks[-1]
#############################################################################################


"""EJ 12/28/2020: This block of code below tosses out any training or test week's that do no include 
5 trading days for a given week. However, I have allowed non-full 5 day trading weeks to be used to
generate class labels for 5 trading day weeks."""
###############################################################################################
train_features_length_regularized = []
train_labels_length_regularized = []
test_features_length_regularized = []
test_labels_length_regularized = []

for x in range(0,len(train_weeks)):
    if len(train_weeks[x]) == 5:
        train_features_length_regularized.append(train_weeks[x])
        train_labels_length_regularized.append(train_labels[x])

for x in range(0,len(test_weeks)):
    if len(test_weeks[x]) == 5:
        test_features_length_regularized.append(test_weeks[x])
        test_labels_length_regularized.append(test_labels[x])

###################################################################################################

# Not necessary but for readability I placed the length trading week length standardized 
# training and test arrays into arrays with easy and uninformative names.
X_train = np.array(train_features_length_regularized)
Y_train= np.array(train_labels_length_regularized)
X_test = np.array(test_features_length_regularized)
Y_test = np.array(test_labels_length_regularized)

""" df_train_saveout = pd.DataFrame({'M':X_train[:,0], 'T':X_train[:,1], 'W':X_train[:,2], 'Th':X_train[:,3], \
    'F':X_train[:,4], 'label':Y_train})
df_train_saveout.to_csv("C:\\Users\\Rachel\\Documents\\GitRepo\\s&p500_20year_daily_price_data\\df_train_all_saveout.csv")

df_test_saveout = pd.DataFrame({'M':X_test[:,0], 'T':X_test[:,1], 'W':X_test[:,2], \
    'Th':X_test[:,3], 'F':X_test[:,4], 'label':Y_test})
df_test_saveout.to_csv("C:\\Users\\Rachel\\Documents\\GitRepo\\s&p500_20year_daily_price_data\\df_test_all_saveout.csv") """

#print(df_train_saveout.head(10))
#print(X_train[0:10])


zipped = list(zip(X_train,Y_train))

train_perm = np.random.permutation(zipped)
sample = train_perm[0:len(train_perm)]
X_train = np.array([i for i,j in sample])
Y_train = np.array([j for i,j in sample])



X_train = np.ascontiguousarray(X_train, dtype=np.float32)


zipped = list(zip(X_test,Y_test))
test_perm = np.random.permutation(zipped)
sample = test_perm[0:len(test_perm)]
X_test = np.array([i for i,j in sample])
Y_test = np.array([j for i,j in sample])



X_test = np.ascontiguousarray(X_test, dtype=np.float32)







# Simple check of proportion of test week arrays that have class label 1.
mean = 0
for x in Y_test:
    mean = mean + x
if len(Y_test) > 0:
        mean = mean/len(Y_test)
else:
    mean = "length of Y_test was 0"
print(mean, len(X_train), len(Y_train), len(X_test), len(Y_test))
############################

# EJ 12/29/2020: Just some stuff I was printing off in getting this set up.
""" print(len(train_features_length_regularized), len(train_labels_length_regularized))
print(len(test_features_length_regularized), len(test_labels_length_regularized), 
len(test_features_length_regularized)*505) """

# EJ 12/28/2020: Feature train and test set generation time and number of training examples.
t1 = time.time()
print("Total Time for Feature Generation: ", t1 - t0)
print("Number of Stored Features: ", len(X_train))

#featrure_dataframe = pd.DataFrame({'Features':})

#########################################################################

""" #J 12/29/2020: The block below is the Support Vector Machine model build. It is not executing in a 
timely manner on the full S&P yet."""
# First attempt at SVM
''' from sklearn import svm
clf = svm.SVC(cache_size = 1000, class_weight = 'balanced', random_state = 0)
clf.fit(X_train, Y_train)
Y_score = clf.decision_function(X_test) '''


""" results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std())) """



''' Y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))
#print(Y_pred, Y_test)

from sklearn.metrics import recall_score
recall = recall_score(Y_test, Y_pred, average='binary')
print('Recall: %.3f' % recall)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(Y_test, Y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision)) '''

""" from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(clf, X_test, Y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision)) """
#plt.savefig('Precision-Recall Curve.png')
#plt.savefig('Precision-Recall Curve.pdf')

t2 = time.time()
print("Total Modeling Time: ", t2 - t1)
print("Total Run Time: ", t2 - t0)
#####################################################################

import rfflearn.cpu as rfflearn                     # Import our module
X = X_train  # Define input data
y = Y_train                          # Defile label data
''' gpc = rfflearn.RFFGPC().fit(X, y)                   # Training
print(gpc.score(X_test, Y_test))                                    # Inference (on CPU)
Y_pred = gpc.predict(X_test) '''

svc = rfflearn.RFFSVC().fit(X, y)                   # Training
print(svc.score(X_test, Y_test) )                                    # Inference (on CPU)
Y_pred = svc.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(Y_test, Y_pred)
print(confusion_mat)







