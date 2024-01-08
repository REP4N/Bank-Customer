import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import seaborn as sns 
import pickle 

#import model 
svm = pickle.load(open('SVC.pkl','rb'))

#load dataset
data = pd.read_csv('Bank Customer Churn Dataset.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Bank Customer')

html_layout1 = """
<br>
<div style="background-color:blue ; padding:2px">
<h2 style="color:white;text-align:center;font-size:40px"><b>Pelanggan Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['SVM','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pelanggan')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Bank Customer</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    

#train test split
y = data['churn']
X = data[['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']]
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    customer_id = st.sidebar.slider('Customer ID', 0, 20, 1)
    credit_score = st.sidebar.slider('Credit Score', 0, 200, 108)
    country = st.sidebar.slider('Country', 0, 140, 40)
    gender = st.sidebar.slider('Gender', 0, 100, 25)
    age = st.sidebar.slider('Age', 21, 100, 24, step=1)
    tenure = st.sidebar.slider('Tenure', 0, 80, 25)
    
    # Menggunakan number_input untuk mendapatkan input float
    balance = st.sidebar.number_input('Balance', min_value=0.0, max_value=2.5, step=0.01, value=0.45)
    
    products_number = st.sidebar.slider('Products Number', 21, 100, 24, step=1)
    credit_card = st.sidebar.slider('Credit Card', 0, 100, 24, step=1)
    active_member = st.sidebar.slider('Active Member', 0, 100, 24, step=1)
    estimated_salary = st.sidebar.slider('Estimated Salary', 0, 100, 24, step=1)
    
    user_report_data = {
        'credit_score': credit_score,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card,
        'active_member': active_member,
        'estimated_salary': estimated_salary
    }
    
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


#Data Pasion
user_data = user_report()
st.subheader('Data Pelangan')
st.write(user_data)

user_result = svm.predict(user_data)
svc_score = accuracy_score(y_test,svm.predict(X_test))

#output
st.subheader('Hasil Prediksi:')
if user_result[0] == 0:
    output = 'Pelanggan tetap (churn negatif)'
else:
    output = 'Pelanggan berpindah (churn positif)'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(svc_score*100)+'%')


