import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

st.title('Profit Prediction')
df=pd.read_csv('modified superstore data.csv')
df

# model creation
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, train_size = 0.85, test_size = 0.15, random_state = 1)

from sklearn.linear_model import LinearRegression
X_train =df_train[['Sales','Quantity','Discount','First Class','Same Day',
             'Second Class','Standard Class','Consumer','Corporate','Home Office','Central','East','South',
             'West','Furniture','Office Supplies','Technology']] # independent variable

Y_train = df_train['Profit'] # dependent variable
lr = LinearRegression()
model = lr.fit(X_train, Y_train)

# --------------------------------------------------------------------------------------------------------

st.title("Prediction")
with st.form(key="my_form1"):
    Sales=st.number_input("Enter sales value:")
    Quantity=st.number_input("Enter quantity:")
    Discount=st.number_input("Enter discount:")
    First_class=st.number_input("Enter first class:")
    Same_day=st.number_input("Enter same day:")
    Second_class=st.number_input("Enter second class:")
    Standard_class=st.number_input("Enter standard class:")
    Consumer=st.number_input("Enter consumer:")
    Corporate=st.number_input("Enter corporate:")
    Home_office=st.number_input("Enter home office:")
    Central=st.number_input("Enter central:")
    East=st.number_input("Enter east:")
    South=st.number_input("Enter south:")
    West=st.number_input("Enter west:")
    Furniture=st.number_input("Enter furniture:")
    Office_supply=st.number_input("Enter office supply:")
    Technology=st.number_input("Enter technology:")
    df=pd.DataFrame({
        'Sales':[Sales],
        'Quantity':[Quantity],
        'Discount':[Discount],
        'First Class':[First_class],
        'Same Day':[Same_day],
        'Second Class':[Second_class],
        'Standard Class':[Standard_class],
        'Consumer':[Consumer],
        'Corporate':[Corporate],
        'Home Office':[Home_office],
        'Central':[Central],
        'East':[East],
        'South':[South],
        'West':[West],
        'Furniture':[Furniture],
        'Office Supplies':[Office_supply],
        'Technology':[Technology]
    })
    if st.form_submit_button("calculate"):
        df
        output=model.predict(df)
        st.write("Profit: ", output)


