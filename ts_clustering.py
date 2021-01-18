import pandas as pd
from pandas import concat
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
t0 = time.time()
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


train_labels = []
test_labels = []

train_weeks = []
test_weeks = []
df_all = pd.DataFrame()
df_all_train = pd.DataFrame()
df_all_test = pd.DataFrame()

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
    sql_query =  """select symbol, date, adjclose_val from  stockmarket_dailyvalues
    where date between '2019-12-20' and '2020-12-21' """
    
    df_all = pd.read_sql_query(sql_query, conn)

except:
    print("Something went wrong with the query.")

#df_all['Weekday'] = df_all['date'].dt.weekday
#df_train = pd.DataFrame(df_all.loc[(df_all['date'] >= '2020-01-01') & (df_all['date'] <= '2020-12-31')])
df_train = pd.DataFrame(df_all)
#df_test = pd.DataFrame(df_all.loc[df_all['date'] >= '2018-01-01'])

df_train['adjclose_pct'] = df_train['adjclose_val'].pct_change().fillna(0.0)
print(type(df_train['adjclose_pct'].values))
#df_test['adjclose_pct'] = df_test['adjclose_val'].pct_change().fillna(0.0)

df_train_grouped = df_train.groupby(['symbol'])
groups = [(group.set_index('date', drop = True))['adjclose_pct'].values \
    for name, group in df_train_grouped]
lengths = []
lengths_length_adj = []
group_length_adj = []
for group in groups:
    lengths.append(len(group))

for x in range(0,len(groups)):
    if len(groups[x]) >= 100:
        group_length_adj.append(groups[x])
        lengths_length_adj.append(lengths[x])

print(min(lengths_length_adj), max(lengths_length_adj))

""" from dtaidistance import dtw
distance = dtw.distance_matrix_fast(ripser_input)
print(distance) """

###########################
# Because I can't help myself...
""" import Cython
import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

def dist(ts1,ts2):
    return dtw.distance_fast(ts1,ts2)

dgms = ripser.ripser(ripser_input,maxdim = 0, distance_matrix = False, do_cocycles = True, \
     metric = dist)['dgms']  

print(dgms)
plot_diagrams(dgms, show=True) """

##############################################
""" from dtaidistance import dtw
from dtaidistance import clustering
from dtaidistance import dtw_visualisation as dtwvis
series = np.ascontiguousarray(group_length_adj, dtype=np.float32)
train_perm = np.random.permutation(series)
sample = train_perm[0:int((0.08)*len(train_perm))]
series = sample
# Custom Hierarchical clustering
model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
cluster_idx = model1.fit(series)
# Augment Hierarchical object to keep track of the full tree
#model2 = clustering.HierarchicalTree(model1)
#cluster_idx = model2.fit(series)
# SciPy linkage clustering
model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {}) """



import seaborn as sns
import scipy
from scipy import stats
from dtaidistance import dtw, clustering
import matplotlib.pyplot as plt

from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
path = dtw.warping_path(s1, s2)
dtwvis.plot_warping(s1, s2, path, filename="warp.png")

timeseries = group_length_adj[0:150]

# Custom Hierarchical clustering
model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
cluster_idx = model1.fit(timeseries)
# Keep track of full tree by using the HierarchicalTree wrapper class
model2 = clustering.HierarchicalTree(model1)
cluster_idx = model2.fit(timeseries)
# SciPy linkage clustering
model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {})
cluster_idx = model3.fit(timeseries)
model3.plot("myplot.png")

