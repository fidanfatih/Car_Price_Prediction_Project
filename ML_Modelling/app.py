# streamlit run app.py

import streamlit as st
import pickle
import pandas as pd

html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Car Price Prediction ML App(Demo)</h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)

# images
from PIL import Image
im = Image.open("image.png")
st.image(im, width=700)

# Categorical Features: ['body_type','Gearing_Type', 'make_model']
# Numerical Features: ['hp', Combumption_Comb', 'Age', 'ss_Xenon headlights', 'ss_LED Headlights', 'km', 'Displacement', 'Gears']

body_type = ['Sedans', 'Station wagon', 'Compact', 'Coupe', 'Van', 'Off-Road', 'Convertible', 'Transporter']
gearing_type = ['Automatic', 'Manual', 'Semi-automatic']
make_model = ['Audi A1', 'Audi A2', 'Audi A3', 'Opel Astra', 'Opel Corsa','Opel Insignia', 'Renault Clio', 'Renault Duster','Renault Espace']

st.sidebar.markdown("## Enter the Features of Your Car")
body_type = st.sidebar.radio("Body Type",(body_type))
gearing_type = st.sidebar.radio("Gearing Type",(gearing_type))
make_model = st.sidebar.radio("Make Model",(make_model))

hp = st.sidebar.slider("HP",40,300,80, step=1)
age = st.sidebar.slider("Age",0,4,1,step=1)
km = st.sidebar.slider("Km",0,350000,45000,step=5000)
gears = st.sidebar.slider("Gears",5,8,7,step=1)
displacement = st.sidebar.slider("Displacement",890,3000,998,step=1)
combumption_comb = st.sidebar.slider("Combumption Comb",3.0,10.0,4.5,step=0.1)

xenon_headlights =1 if st.sidebar.checkbox('Xenon Headlights') else 0
led_headlights =1 if st.sidebar.checkbox('LED Headlights') else 0

my_dict = {'hp':hp,
           'Age':age,
           'make_model':make_model,
           'body_type':body_type,
           'Displacement':displacement,
           'Gears':gears,
           'ss_Xenon headlights':xenon_headlights,
           'Combumption_Comb':combumption_comb,
           'km':km,
           'Gearing_Type':gearing_type,
           'ss_LED Headlights':led_headlights}
df_table = pd.DataFrame.from_dict([my_dict])

# Table
df_table2=df_table.rename(columns={'hp':'HP',
                          'make_model':'Make Model',
                          'body_type':'Body Type',
                          'Displacement':'Displacement',
                          'Gears':'Gears',
                          'ss_Xenon headlights':'Xenon Headlights',
                          'Combumption_Comb':'Combumption Comb',
                          'km':'KM',
                          'Gearing_Type':'Gearing Type',
                          'ss_LED Headlights':'Led Headlights'})
st.table(df_table2) 


all_columns=pd.read_csv("final_scout_20201204.csv").drop('price',axis=1).columns
df = pd.get_dummies(df_table).reindex(columns=all_columns, fill_value=0)

st.subheader("Choose ML Model:")
model = st.radio('',['XGBoost Regressor', 'Random Forest Regressor'])
# Button
xgb_model = pickle.load(open("XGBoostReg.pkl","rb"))
rfr_model = pickle.load(open("RFReg.pkl","rb"))

if st.button("Submit"):
    import time
    with st.spinner("ML Model is loading..."):
        my_bar=st.progress(0)
        for p in range(0,101,10):
            my_bar.progress(p)
            time.sleep(0.05)

        if model=='Random Forest Regressor':
            prediction= rfr_model.predict(df)
        elif model=='XGBoost Regressor':
            prediction= xgb_model.predict(df)

    st.success(f"The Estimated Price of Your Car is €{int(prediction[0])}")
