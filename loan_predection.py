
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# Read the train_loan_predection.csv into pandas DataFrame
loan_df = pd.read_csv('./data/loan/train_loan_predection.csv')
loan_df = loan_df.set_index('Loan_ID')
loan_df = loan_df.dropna(how='any')

# Convert Dependents column to integers
loan_df['Dependents'] = loan_df.Dependents.apply(lambda x: int(x.replace('+', '')))

# ## Addition of more fields

# In[44]:

loan_status_map = {'Y': 1, 'N': 0}
loan_df['Loan_Status_int'] = loan_df.Loan_Status.map(loan_status_map)
loan_df['TotalIncome'] = loan_df.ApplicantIncome + loan_df.CoapplicantIncome
loan_df[loan_df.Loan_Status == 'Y'].TotalIncome.plot(kind='hist', bins=20)
columns_retained = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                    'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']

columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
df = loan_df[columns_retained]
df = pd.get_dummies(df, columns=columns, drop_first=True)

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
y_train = X_train['Loan_Status_Y']
y_test = X_test['Loan_Status_Y']
X_train = X_train.drop('Loan_Status_Y', axis=1)
X_test = X_test.drop('Loan_Status_Y', axis=1)
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, max_features=None)
clf.fit(X_train, y_train)
d = dict(zip(clf.feature_importances_, X_test.columns))
for k in sorted(d.keys(), reverse=True):
    print(d[k])

y_pred = clf.predict(X_test)
print(confusion_matrix(y_pred=y_pred, y_true=y_test))
print(accuracy_score(y_pred=y_pred, y_true=y_test))

df_test = pd.read_csv('./data/loan/test_loan_predection.csv')
df_test = df_test.set_index('Loan_ID')
df_test = df_test.dropna(how='any')
df_test['Dependents'] = df_test.Dependents.apply(lambda x: int(x.replace('+', '')))

columns_retained = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                    'Loan_Amount_Term', 'Credit_History', 'Property_Area']

columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
df_test = df_test[columns_retained]
df_test = pd.get_dummies(df_test, columns=columns, drop_first=True)

y_test_pred = clf.predict(df_test)
df_test['Loan_Status'] = y_test_pred

df_test.reset_index()[['Loan_ID', 'Loan_Status']].to_csv('test_submission.csv', index=False)
