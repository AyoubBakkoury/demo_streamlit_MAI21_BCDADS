#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:28:30 2021

@author: ayoub
"""

import streamlit as st 
import pandas as pd 
from sklearn.model_selection import train_test_split 

from preprocess import preprocess , modeling


df = pd.read_csv('/Users/ayoub/Documents/GitHub/demo_streamlit_MAI21_BCDADS/train.csv')

pages= ['Présentation', 'Modélisation']

partie  = st.sidebar.radio('Parties', pages)


if partie == pages[0]:
    
    ##présentation 
    st.title('Données Titanic')
    st.write(df.head())


else: 

    st.title('Modélisation')


    X, y = preprocess()

#st.write(X.isna().sum())




    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)



    options = ['Regression log', 'KNN']

    choix = st.radio('Choisissez un modèle', options)
    
    score = modeling(choix, X_train, y_train, X_test, y_test)
 
    
    st.write('Score pour', choix, ':', score)  
    
    
    
    
    
    
    
    
    