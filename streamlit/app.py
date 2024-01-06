import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

st.title('Text Cassification')
df = pd.read_csv('cleaned_bbc_data.csv')
df
#----------------------------------------------------------model creation start ---------------------------------------------
# Training model
from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()

vectorizer = TfidfVectorizer()
X = df['text']
Y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15) #Splitting dataset

# #Creating Pipeline
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=2000)),
                     ('clf', LogisticRegression())])
# #Training model
model = pipeline.fit(X_train, y_train)

#----------------------------------------------------------model creation end ---------------------------------------------
# file = open('news.txt','r')
# news = file.read()
# file.close()

st.title("Prediction")

news=st.text_area("Text to translate:")
if st.button("Submit"):
    news_data = {'predict_news':[news]}
    news_data_df = pd.DataFrame(news_data)
    # news_data_df
    predict_news_cat = model.predict(news_data_df['predict_news'])
    st.write("Predicted news category = ",predict_news_cat[0])
else:
    st.write("please enter the news")

# st.code('print("Hello World)')
    
# ### Form
with st.form(key="my_form"):
    username=st.text_input("Enter username:")
    password=st.text_input("Enter password:")
    st.form_submit_button("login")
if st.button("login"):
    st.write("hello world")


# important 
    # sakesama hamle streamlit ko lagi model train nagarne yo file ma 
    # yesma hamle data present garna matra use garne

st.title("diabetes")
data=pd.read_csv("diabetes2.csv")
data
X=data.iloc[:,:8]
Y=data.iloc[:,8:]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)

with st.form(key="my_form1"):
    Pregnancies=st.number_input("Enter pregnency:")
    Glucose=st.number_input("Enter glucose:")
    BloodPressure=st.number_input("Enter blood pressure:")
    SkinThickness=st.number_input("Enter skin thickness:")
    Insulin=st.number_input("Enter insulin:")
    BMI=st.number_input("Enter BMI:")
    DiabetesPedigreeFunction=st.number_input("Enter diabetes pedigree function:")
    Age=st.number_input("Enter age:")
    df=pd.DataFrame({
        'Pregnancies':[Pregnancies],
        'Glucose':[Glucose],
        'BloodPressure':[BloodPressure],
        'SkinThickness':[SkinThickness],
        'Insulin':[Insulin],
        'BMI':[BMI],
        'DiabetesPedigreeFunction':[DiabetesPedigreeFunction],
        'Age':[Age]
    })
    if st.form_submit_button("calculate"):
        df
        output=lr.predict(df)
        output=output[0][0]
        st.write("Outcome: ", output)