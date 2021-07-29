# Data-Science-Portfolio-Scripts
In this repository lives some of the code I have produced.
- credit_fraud_box.ipynb 
- stock_ts_feature_slicing.py
- svm_momentum_model.py
- ts_clustering.py

# Data Wrangling and Manipulation Folder
In this data wrangling and manipulation folder you will find examples of python calling sql in notebooks and scripts and a saved sql script that I used in a Hacker Rank challenge recently to refreshen some of my SQL skills. I have also included an insertion into a postgres database that I made as part of personal project earlier this year.

# credit_fraud_box.ipynb
This is a notebook for resampling the 15 features that are most well correlated with the class labels and then training and XGBoost model for classification. The resampling is a combinatino of under- and over-sampling that is accomplished with the imblearn package in python. I have writen a medium article about this project. This notebook will reproduce the results quoted in that medium article.
      
