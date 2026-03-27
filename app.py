import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

# 1. SETUP THE PAGE
st.set_page_config(page_title="TrendScout AI", page_icon="📈")
st.title("📈 TrendScout AI")
st.write("Predict if a stock or crypto will go UP or DOWN tomorrow.")

# 2. CREATE THE SIDEBAR INPUTS
ticker = st.sidebar.text_input("Enter Ticker (e.g., AAPL, TSLA, BTC-USD)", "AAPL")
button = st.sidebar.button("Analyze")

# 3. WHAT HAPPENS WHEN YOU CLICK BUTTON
if button:
    with st.spinner("Fetching data and calculating..."):
        try:
            # Fetch data from Yahoo Finance
            data = yf.download(ticker, period="1y")
            
            if data.empty:
                st.error("Could not find that ticker. Try another one.")
            else:
                # SHOW RAW DATA
                st.subheader(f"Historical Data for {ticker}")
                st.line_chart(data['Close'])

                # AI LOGIC (Linear Regression)
                # We create a simple "Day" column (1, 2, 3...) to predict price
                data = data.reset_index()
                data['Day'] = np.arange(len(data))
                
                X = data[['Day']] # Input: The day number
                y = data['Close']  # Output: The price
                
                # Train the AI brain
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict tomorrow (which is the next day number)
                tomorrow_day = np.array([[len(data)]])
                prediction = model.predict(tomorrow_day)[0]
                
                # Get the last actual price to compare
                last_price = data['Close'].iloc[-1]
                
                # SHOW RESULTS
                st.subheader("AI Prediction")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Last Price", f"${last_price:.2f}")
                
                with col2:
                    st.metric("Predicted Tomorrow", f"${prediction:.2f}")
                
                # DECISION LOGIC
                if prediction > last_price:
                    st.success("✅ Trend: UP (Buy Signal)")
                else:
                    st.error("📉 Trend: DOWN (Sell Signal)")

        except Exception as e:
            st.error("Something went wrong. Check your internet connection.")
