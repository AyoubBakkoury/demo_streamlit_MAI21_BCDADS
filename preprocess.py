#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:59:03 2021

@author: ayoub
"""

#Preprocessing 

import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def preprocess(): 
    df = pd.read_csv('/Users/ayoub/Documents/GitHub/demo_streamlit_MAI21_BCDADS/train.csv')



    X = df.drop('Survived', axis = 1)

    y = df['Survived']

    X = X.drop(['Name', 'PassengerId', 'Cabin', 'Ticket'], axis = 1)
    X['Age'] = X['Age'].fillna(X['Age'].mode()[0])
    X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
    X['Sex']= X['Sex'].apply( lambda x : x== 'female')

    X= pd.get_dummies(X, columns = ['Embarked'])

    return X, y 


def modeling(choix, X_train, y_train, X_test, y_test):

    options = ['Regression log', 'KNN']    
    if choix == options[0]: 
    
        model = LogisticRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
    
    
    if choix == options[1]: 
    
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)   
    return score 
    
        
        