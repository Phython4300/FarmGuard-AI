import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st

def load_price_data(file_path="cassava_price.csv"):
    df = pd.read_csv(file_path)
    df.rename(columns={"date": "ds", "price": "y"}, inplace=True)
    return df

def predict_prices(df, periods=30):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast, model

def plot_forecast(model, forecast):
    fig = model.plot(forecast)
    st.pyplot(fig)
