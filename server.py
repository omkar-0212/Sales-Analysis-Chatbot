import pandas as pd
import streamlit as st
import plotly.express as px
import pyttsx3
import tempfile
import os
import requests
import threading
import openai
import os
from scipy.io.wavfile import write
import sounddevice as sd
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv

# OpenAI API Key
api_key = st.secrets["api_key"]
api_url = st.secrets["api_url"]

# Initialize the pyttsx3 engine for text-to-speech
engine = pyttsx3.init()


def record_audio(duration=5, sample_rate=16000):
    """Records audio from the microphone for the given duration."""
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    return audio, sample_rate


def save_audio_to_file(audio, sample_rate):
    """Saves the recorded audio to a WAV file."""
    temp_filename = tempfile.mktemp(suffix='.wav')
    write(temp_filename, sample_rate, audio)
    return temp_filename


def transcribe_audio(file_path):
    """Transcribes the audio file using OpenAI Whisper."""
    headers = {'Authorization': f'Bearer {api_key}'}
    with open(file_path, 'rb') as f:
        response = requests.post(api_url, headers=headers, files={'file': f},
                                 data={'model': 'whisper-1', 'language': 'en'})
    if response.status_code == 200:
        result = response.json()
        return result.get('text', 'No transcription text found.')
    return None


def speak(text):
    """Speaks the given text using pyttsx3 in a separate thread."""

    def run_speech():
        local_engine = pyttsx3.init()  # Reinitialize the engine for each call
        local_engine.setProperty('rate', 150)  # Set speaking speed
        local_engine.setProperty('volume', 1.0)  # Set volume (0.0 to 1.0)
        local_engine.say(text)
        local_engine.runAndWait()
        local_engine.stop()  # Ensure the engine is stopped after speaking

    threading.Thread(target=run_speech).start()


def load_data(file_path):
    """Loads an Excel file containing sales, inventory, and purchasing data."""
    data = pd.read_excel(file_path)
    return data


def analyze_sales(data):
    """Analyzes the sales data and returns a summary of total sales per product."""
    sales_summary = data.groupby('Product')['Sales'].sum().sort_values(ascending=False)
    return sales_summary


def plot_sales(sales_summary):
    """Generates a bar chart for the top-selling products."""
    fig = px.bar(sales_summary, x=sales_summary.index, y=sales_summary.values,
                 labels={'x': 'Product', 'y': 'Total Sales'}, title="Top-Selling Products")
    return fig


def analyze_inventory(data):
    """Analyzes the inventory data and returns a summary of stock levels."""
    inventory_summary = data.groupby('Product')['Stock'].sum().sort_values(ascending=False)
    return inventory_summary


def plot_inventory(inventory_summary):
    """Generates a bar chart for the inventory levels of products."""
    fig = px.bar(inventory_summary, x=inventory_summary.index, y=inventory_summary.values,
                 labels={'x': 'Product', 'y': 'Stock Level'}, title="Inventory Levels")
    return fig


def forecast_sales_for_product(data, product_name):
    """Forecasts future sales for a specific product using the Prophet model."""
    # Filter the data for the specific product
    product_data = data[data['Product'] == product_name][['Date', 'Sales']]
    product_data = product_data.rename(columns={'Date': 'ds', 'Sales': 'y'})

    # Forecast using Prophet
    model = Prophet()
    model.fit(product_data)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast


def plot_forecast(forecast):
    """Generates a plot for the sales forecast."""
    fig = px.line(forecast, x='ds', y='yhat', title='Sales Forecast', labels={'ds': 'Date', 'yhat': 'Forecasted Sales'})
    return fig


def analyze_purchasing(data):
    """Analyzes the purchasing data and returns total purchases per product."""
    purchasing_summary = data.groupby('Product')['Purchases'].sum().sort_values(ascending=False)
    return purchasing_summary


def plot_purchasing(purchasing_summary):
    """Generates a bar chart for purchasing data."""
    fig = px.bar(purchasing_summary, x=purchasing_summary.index, y=purchasing_summary.values,
                 labels={'x': 'Product', 'y': 'Total Purchases'}, title="Purchasing Data")
    return fig


def predict_stockout(data):
    """Predicts stockouts by calculating when stock levels will reach zero."""
    sales_summary = data.groupby('Product')['Sales'].sum()
    stock_summary = data.groupby('Product')['Stock'].sum()
    stockout_days = stock_summary / (sales_summary / 30)  # Assuming 30 days of sales data
    stockout_prediction = stockout_days.sort_values()
    return stockout_prediction


def plot_stockout(stockout_prediction):
    """Generates a bar chart for stockout predictions in days."""
    fig = px.bar(stockout_prediction, x=stockout_prediction.index, y=stockout_prediction.values,
                 labels={'x': 'Product', 'y': 'Days Until Stockout'}, title="Stockout Prediction")
    return fig


def train_sales_model(data):
    """Trains a linear regression model for sales prediction and returns predictions for visualization."""
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(pd.Timestamp.timestamp)  # Convert to timestamp for regression
    X = data[['Date']]
    y = data['Sales']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return model, mse, X_test, y_test, predictions  # Return the test data and predictions



def predict_sales(model, future_dates):
    """Predicts future sales given a trained model and future dates."""
    future_dates = pd.to_datetime(future_dates)
    future_dates = future_dates.map(pd.Timestamp.timestamp).values.reshape(-1, 1)
    return model.predict(future_dates)

def generate_openai_response(prompt):
    """Generates a response from OpenAI based on the given prompt."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'gpt-3.5-turbo',  # Use the latest available model
        'messages': [{'role': 'user', 'content': prompt}]
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return "I'm unable to generate a response at the moment."

def main():
    st.title("TMG Voice-Activated Assistant")
    st.write("Please Speak into the microphone or type your command below.")

    # File upload widget
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    st.warning("Please upload supported files")

    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.write("File uploaded successfully!")

        if 'model' not in st.session_state:
            st.session_state.model = None

        # Manual input text box
        manual_command = st.text_input("Type your command here:")

        if st.button("Submit Command"):
            normalized_command = manual_command.lower()
            st.write(f"Command recognized: **{normalized_command}**")

            # Command processing
            process_command(normalized_command, data)
        elif st.button("Start Listening"):
            with st.spinner("Listening..."):
                audio, sample_rate = record_audio()
                file_path = save_audio_to_file(audio, sample_rate)
                command = transcribe_audio(file_path)
                os.remove(file_path)  # Clean up the temporary file

                # Display the recognized command
                if command:
                    normalized_command = command.lower()
                    st.write(f"Command recognized: **{normalized_command}**")
                    # Command processing
                    process_command(normalized_command, data)
                else:
                    st.write("No command recognized.")


def process_command(normalized_command, data):
    """Processes the recognized command and generates OpenAI response."""
    if 'sales analysis' in normalized_command:
        sales_summary = analyze_sales(data)
        fig = plot_sales(sales_summary)
        st.plotly_chart(fig)
        response_text = "I have completed the sales analysis."
    elif 'inventory analysis' in normalized_command:
        inventory_summary = analyze_inventory(data)
        fig = plot_inventory(inventory_summary)
        st.plotly_chart(fig)
        response_text = "I have completed the inventory analysis."
    elif 'forecast sales' in normalized_command:
        speak("Please tell me the product name for sales forecasting.")
        st.write("Listening for the product name...")

        product_audio, product_sample_rate = record_audio()
        product_file_path = save_audio_to_file(product_audio, product_sample_rate)
        product_name = transcribe_audio(product_file_path)
        os.remove(product_file_path)  # Clean up the temporary file

        if product_name:
            st.write(f"Product recognized: **{product_name}**")
            forecast = forecast_sales_for_product(data, product_name)
            fig = plot_forecast(forecast)
            st.plotly_chart(fig)
            response_text = f"Forecast for product **{product_name}** completed."
        else:
            response_text = "No product name recognized."
    elif 'purchasing analysis' in normalized_command:
        purchasing_summary = analyze_purchasing(data)
        fig = plot_purchasing(purchasing_summary)
        st.plotly_chart(fig)
        response_text = "I have completed the purchasing analysis."
    elif 'stockout predictions' in normalized_command:
        stockout_prediction = predict_stockout(data)
        fig = plot_stockout(stockout_prediction)
        st.plotly_chart(fig)
        response_text = "I have completed the stockout prediction analysis."
    elif 'train sales model' in normalized_command:
        st.session_state.model, mse, X_test, y_test, predictions = train_sales_model(data)

        # Create a DataFrame for plotting
        results_df = pd.DataFrame({
            'Date': pd.to_datetime(X_test.values.flatten(), unit='s'),  # Convert timestamps back to datetime
            'Actual Sales': y_test,
            'Predicted Sales': predictions
        })

        # Plot actual vs. predicted sales
        fig = px.line(results_df, x='Date', y=['Actual Sales', 'Predicted Sales'],
                      labels={'value': 'Sales', 'variable': 'Legend'},
                      title='Actual vs. Predicted Sales')
        st.plotly_chart(fig)

        response_text = f"Sales model trained successfully with MSE: {mse:.2f}."
    elif 'predict future sales' in normalized_command:
        if st.session_state.model:
            speak("Please tell me the future date for sales prediction.")
            st.write("Listening for the future date...")

            date_audio, date_sample_rate = record_audio()
            date_file_path = save_audio_to_file(date_audio, date_sample_rate)
            future_date = transcribe_audio(date_file_path)
            os.remove(date_file_path)  # Clean up the temporary file

            if future_date:
                st.write(f"Future date recognized: **{future_date}**")
                # Predict sales for the future date
                future_sales = predict_sales(st.session_state.model, [future_date])
                response_text = f"Predicted sales for **{future_date}**: {future_sales[0]:.2f}."
            else:
                response_text = "No future date recognized."
        else:
            response_text = "Please train the sales model first."
    else:
        response_text = "Sorry, I didn't understand the command."

    # Generate OpenAI response
    openai_response = generate_openai_response(normalized_command)
    st.write(openai_response)

    st.write(response_text)
    speak(response_text)
    speak(openai_response)


if __name__ == "__main__":
    main()
