import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import pickle
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')


with open('one_hot_encoder_geo.pkl','rb') as file : 
    onehot_encoder_geo = pickle.load(file)
with open('label_encoder_gender.pkl','rb') as file : 
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file : 
    scaler = pickle.load(file)


## streamlit app

st.title('Customer Churns Prediction') 
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of products', 1, 4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])


input_data = {
    'CreditScore' : [credit_score],
    'Geography' : [geography],
    'Gender' : [gender],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]


}

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded = pd.DataFrame(geo_encoded,columns= onehot_encoder_geo.get_feature_names_out(['Geography']))
df_top = pd.DataFrame(input_data)
df_top['Gender_encoded'] = label_encoder_gender.transform(df_top['Gender'])
df = pd.concat([df_top.reset_index(drop=True), geo_encoded], axis=1)
df.drop(columns=['Geography','Gender'],inplace=True)
input_data_scaled = scaler.transform(df)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba>0.5 : 
    st.write('the customer is likely to churn.')
else :
    st.write('the customer is not likely to churn.')
