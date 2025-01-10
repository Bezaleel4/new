# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # THYROID CANCER RECURRANCE DETECTOR MODEL

# ### Import All Necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from flask import Flask, request, jsonify
import streamlit as st
import warnings
warnings.filterwarnings("ignore")





# Uploading the model
model = joblib.load("Cancer Predictor.joblib")

 
st.header("Cancer Predictor")
st.image("cancer.jpg", width = 400)
col1, col2, col3, col4 = st.columns(4)
with col1:
    Age = st.text_input("Age")
    Gender = st.number_input("Gender", min_value = 0, max_value = 1)     
    Smoking = st.number_input("Smoking", min_value = 0, max_value = 1)
    Hx_Smoking = st.number_input("Hx_Smoking", min_value = 0, max_value = 1)
    Hx_Radiothreapy = st.number_input("Hx_Radiothreapy", min_value = 0, max_value = 1)
    
with col2:
    Physical_Examination = st.number_input("Physical_Examination", min_value = 0, max_value = 5)
    Adenopathy = st.number_input("Adenopathy", min_value = 0, max_value = 5)
    Pathology = st.number_input("Pathology", min_value = 0, max_value = 3)
    Focality = st.number_input("Focality", min_value = 0, max_value = 1)
    Risk = st.number_input("Risk", min_value = 0, max_value = 2)
    
    
with col3:
    T = st.number_input("T", min_value = 0, max_value = 6)
    N = st.number_input("N", min_value = 0, max_value = 2)
    M = st.number_input("M", min_value = 0, max_value = 1)
    Stage = st.number_input("Stage", min_value = 0, max_value = 4)
    Response = st.number_input("Response", min_value = 0, max_value = 3)
    
    
with col4:
    Clinical_Hyperthyroidism = st.number_input("Clinical_Hyperthyroidism", min_value = 0, max_value = 1)
    Clinical_Hypothyroidism = st.number_input("Clinical_Hypothyroidism", min_value = 0, max_value = 1)
    Euthyroid = st.number_input("Euthyroid")
    Subclinical_Hyperthyroidism = st.number_input("Subclinical_Hyperthyroidism", min_value = 0, max_value = 1)
    Subclinical_Hypothyroidism = st.number_input("Subclinical_Hypothyroidism", min_value = 0, max_value = 1)

if st.button("Prediction"):
    predict = model.predict([[Age, Gender, Smoking, Hx_Smoking, Hx_Radiothreapy, Physical_Examination, Adenopathy,
                                     Pathology, Focality, Risk, T, N, M, Stage, Response, Clinical_Hyperthyroidism,
                                     Clinical_Hypothyroidism, Euthyroid, Subclinical_Hyperthyroidism, Subclinical_Hypothyroidism]])
    st.success(predict[0])
    st.write("Prediction is 0 cancer is likely not to reoccur, but if 1 cancer is likely to reoccur")



# -






