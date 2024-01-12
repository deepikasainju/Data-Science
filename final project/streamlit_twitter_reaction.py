import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

st.title("Text Classification for twitter training")
df=pd.read_csv("cleaned_twittertrain_data.csv")
df

from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()

vectorizer = TfidfVectorizer() 
X = df['Review'].apply(lambda X: np.str_(X)) #indepedent 
Y = df['Reaction'] #dependent

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15) 

# #Creating Pipeline
pipeline = Pipeline([('vect', vectorizer), 
                     ('chi',  SelectKBest(chi2, k=2000)), 
                     ('clf', LogisticRegression(random_state=1,max_iter=1000))]) 


model = pipeline.fit(X_train, Y_train)


st.title("Reaction Prediction")

react=st.text_area("Text to translate:")
if st.button("Submit"):
    react_data = {'predict_react':[react]}
    react_data_df = pd.DataFrame(react_data)
    predict_react_cat = model.predict(react_data_df['predict_react'])
    st.write("Predicted reaction of review = ",predict_react_cat[0])
else:
    st.write("please enter the news")
