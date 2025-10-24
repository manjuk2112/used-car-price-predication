import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
import os
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("car_data.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'car_data.csv' is in the same directory.")
        return None

# Preprocess data
def preprocess_data(df):
    df = df.dropna()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    
    label_encoders = {}
    for col in ['car_model', 'fuel_type', 'transmission', 'seller_type', 'owner']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Feature scaling
    scaler = StandardScaler()
    numeric_cols = ['year', 'selling_price', 'km_driven']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, label_encoders

def train_or_load_model(df, model_path="car_price_model.pkl"):
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.success("Pre-trained model loaded successfully!")
        return model
    else:
        X = df.drop('selling_price', axis=1)
        y = df['selling_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        st.success(f"Model trained and saved (MAE: ₹{mae:,.2f})")
        return model

def plot_model_price_comparison(df, label_encoders):
    plt.figure(figsize=(14, 8))
    
    car_model_names = label_encoders['car_model'].classes_
    df['car_model_name'] = df['car_model'].map(lambda x: car_model_names[x])
    avg_prices = df.groupby('car_model_name')['selling_price'].mean().sort_values()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(avg_prices)))
    bars = plt.barh(avg_prices.index, avg_prices.values, color=colors)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'₹{width:,.0f}',
                va='center', ha='left', fontsize=9)
    
    plt.xlabel('Average Selling Price (₹)', fontsize=12)
    plt.ylabel('Car Model', fontsize=12)
    plt.title('Price Comparison Across Car Models', fontsize=14, pad=20)
    plt.grid(axis='x', alpha=0.3)
    st.pyplot(plt)
    plt.clf()

def main():
    st.set_page_config(layout="wide")
    st.title("🚗 Car Price Prediction Dashboard")
    
    df = load_data()
    
    if df is not None:
        df, label_encoders = preprocess_data(df)
        model = train_or_load_model(df)
        
        st.header("📈 Comprehensive Price Analysis")
        plot_model_price_comparison(df, label_encoders)
        
        st.header("🔍 Price Prediction")
        col1, col2 = st.columns(2)
        
        with col1:
            car_model = st.selectbox("Car Model", label_encoders['car_model'].classes_)
            year = st.slider("Year", 2000, 2025, 2015)
            km_driven = st.slider("Mileage (km)", 0, 300000, 50000, 5000)
            
        with col2:
            fuel_type = st.selectbox("Fuel Type", label_encoders['fuel_type'].classes_)
            transmission = st.selectbox("Transmission", label_encoders['transmission'].classes_)
            seller_type = st.selectbox("Seller", label_encoders['seller_type'].classes_)
            owner = st.selectbox("Owner History", label_encoders['owner'].classes_)
        
        if st.button("Estimate Price"):
            try:
                # Prepare input data
                input_data = np.array([[year, 
                                      km_driven,
                                      label_encoders['car_model'].transform([car_model])[0],  
                                      label_encoders['fuel_type'].transform([fuel_type])[0],
                                      label_encoders['transmission'].transform([transmission])[0],
                                      label_encoders['seller_type'].transform([seller_type])[0],
                                      label_encoders['owner'].transform([owner])[0]]])
                
                # Make prediction
                if model:
                    prediction = model.predict(input_data)
                    st.success(f"Estimated Price: ₹{prediction[0]:,.2f}")
                    
                    # Visualize prediction
                    plt.figure(figsize=(8, 2))
                    plt.barh(['Predicted Price'], [prediction[0]], color='skyblue')
                    plt.xlabel('Price (₹)')
                    plt.tight_layout()
                    st.pyplot(plt)
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
