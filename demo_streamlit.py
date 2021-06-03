#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:28:30 2021

@author: ayoub
"""

import streamlit as st 
import pandas as pd 

st.title('Demo Streamlit')

df = pd.read_csv('/Users/ayoub/Documents/GitHub/demo_streamlit_MAI21_BCDADS/train.csv', index_col = 'PassengerID')