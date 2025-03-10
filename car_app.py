import streamlit as st
import requests
import pickle
import pandas as pd
import numpy as np

st.header('Welcome to Car Prediction 🚗')

model = pickle.load(open('LinearRegressionModel_Car_price_prediction.pkl','rb'))
data = pickle.load(open('data_car_clean.pkl', 'rb'))
x_test_set = pickle.load(open('x_test_set.pkl','rb'))

ans_pre = []
# name , company, year, kms_driver,fuel_types

def main():

    # name , company, year, kms_driver,fuel_types
    

    car_list_model = st.selectbox("Select Car Model", data['name'].unique())

    car_list_company = st.selectbox("Select Car Company", data['company'].unique())
    car_list_year = st.selectbox("Select Car Year", sorted(data['year'].unique(), reverse=True))

    car_list_km = st.number_input("Enter Kilometers Driven", min_value=1000, max_value=500000, step=1000)

    car_list_feul = st.selectbox("Select Fuel Type", data['fuel_type'].unique())

   
    if st.button('Prediction', key='predict'):

      try:
         input_data = pd.DataFrame({
            'company': [car_list_company],
            'name': [car_list_model],
            'year': [car_list_year],
            'fuel_type': [car_list_feul],
            'kms_driven': [car_list_km]
         })
         prediction = model.predict(input_data)
         output = round(prediction[0][0], 2)
         st.write(f'The car prices is: {output} $')
         
      except Exception as e:
         st.error(f"Something went wrong: {e} ")

if __name__ == "__main__":
    main()
      
    


