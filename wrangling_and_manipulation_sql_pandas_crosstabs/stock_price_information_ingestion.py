import datetime
import psycopg2
import psycopg2.extras
import os
import sys
from dateutil import parser
import csv
from unidecode import unidecode
import configparser

ini_file = str(sys.argv[1])
ticker_symbol = str(sys.argv[2]).replace(".csv", '')
config = configparser.ConfigParser()
config.sections()
config.read(ini_file)

#data_file = str(config['Directory']['datafile'])
data_file = str(sys.argv[2])
local_path = str(config['Directory']['localpath'])
db_dbname = str(config['Database']['db_dbname'])
db_user = str(config['Database']['db_user'])
db_password = str(config['Database']['db_password'])
db_host = str(config['Database']['db_host'])
csv_file_path = (os.path.join(local_path, data_file))
num_inserts=10000

#print("password: " + str(db_password))
#print("path: " + str(local_path))
#print("dbname: " + str(db_dbname))
#print("db username: "+ str(db_user))
print("file: " + str(data_file))

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


i = 0
infileHandle = open(csv_file_path, 'r')
print("starting to read csv")

row_list = []
for row in csv.reader(infileHandle, quotechar='"', delimiter = ',', quoting=csv.QUOTE_ALL):
    if i > 0:
        symbol = ticker_symbol,
        date = str(row[0])
        opened_val = str(row[1])
        high_val = str(row[2])
        low_val = str(row[3])
        closed_val = str(row[4])
        adjclose_val = str(row[5])
        volume = str(row[6])
        row_list.append([symbol, date, opened_val, high_val, low_val, closed_val, adjclose_val, volume])
    i = i + 1

infileHandle.close()

sql = """INSERT INTO stockmarket_dailyvalues (
    symbol, date, opened_val, high_val, low_val, closed_val, adjclose_val, volume
)

values (%s, %s, %s, %s, %s, %s, %s, %s)

"""

try:
    psycopg2.extras.execute_batch(cur, sql, row_list)
    conn.commit()
    print
    now = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print(str(num_inserts) + "rows commited to " + str(db_dbname) + " at " + now)

except:
    print
    print("Unexpected Error")
    print(str(sys.exc_info()[0]))
    print(str(sys.exc_info()[1]))
    print(str(sys.exc_info()[2]))
    print

conn.commit()
conn.close()
#print("end strock price data ingestion")