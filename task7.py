import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from sklearn.svm import SVC

# Preprocess Titanic Data
def preprocess_titanic(df):

    # Seperate categorical data into binary labels
    #   PClass into lower middle upper
    df['Lower'] = df['Pclass'].copy().replace(1, 0).replace(2, 0).replace(3, 1)
    df['Middle'] = df['Pclass'].copy().replace(1, 0).replace(2, 1).replace(3,0)
    df['Upper'] = df['Pclass'].copy().replace(1, 1).replace(2, 0).replace(3, 0)
    #   Sex into female and male
    df['Female'] = df['Sex'].copy().replace('female', 1).replace('male', 0)
    df['Male'] = df['Sex'].copy().replace('female', 0).replace('male', 1)
    #   Embarkment to location binary labels
    df['Cherbourg'] = df['Embarked'].copy().replace('C', 1).replace('Q',0).replace('S',0)
    df['Queenstown'] = df['Embarked'].copy().replace('C', 0).replace('Q',1).replace('S',0)
    df['Southampton'] = df['Embarked'].copy().replace('C', 0).replace('Q',0).replace('S',1)
    # Delete columns we are not using
    del df['Name']
    del df['Ticket']
    del df['Sex']
    del df['Cabin']
    del df['Embarked']
    del df['PassengerId']
    del df['SibSp']
    del df['Parch']
    del df['Pclass']
    # Pop survived if in it, and return it.  (gets training labels)
    if 'Survived' in df.columns:
        lbl = df.pop('Survived')
    else:
        lbl = None
    # Return data ready for imputation & then the label set
    return df, lbl

# KNN Impute the data based off training data
def impute(train, test):
    imputer = KNNImputer(missing_values=np.nan)
    imputer.fit(train.to_numpy())
    train_i = imputer.transform(train.to_numpy())
    test_i = imputer.transform(test.to_numpy())
    return train_i, test_i

print('\nLoading & Preprocessing the titanic data')

# Load in titanic Data
train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')

# Preprocess the data using the preprocess function defined above
train, train_lbl = preprocess_titanic(train)
test, test_lbl = preprocess_titanic(test)
# Impute the data using the imputation function
train_imputed, test_imputed = impute(train, test)
# Replace the null values with the imputed values
train[train.isnull()] = train_imputed
test[test.isnull()] = test_imputed
# Calculate correlations between the data and the labels
print('\nCalculating Correlations')
train.corrwith(train_lbl).to_csv('results/corr.csv')

for k in ['linear','poly','rbf']:
    # Create and test decision tree
    print(f'\n{k}')
    sup_vec_mac = SVC(kernel=k,degree=2)
    sup_vec_mac.fit(train, train_lbl)
    # Evaluate
    print("Score: ",sup_vec_mac.score(train, train_lbl))
    runs = cross_val_score(sup_vec_mac, train, train_lbl, cv=5)
    print("Avg Cross Val Score: ", sum(runs)/5)
    print("Max Cross Val Score: ", max(runs))
    results = sup_vec_mac.predict(test)
