import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
np.float_ = np.float64
import plotly.express as px
import pyttsx3
import tempfile
import os
from scipy.io.wavfile import write
import requests
from prophet import Prophet
import sounddevice as sd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
import matplotlib.pyplot as plt
import seaborn as sns
# OpenAI API Key (Replace 'your-api-key' with your actual OpenAI API key)
api_key = 'AIzaSyBvJisDs5TyTaOy8vrVkc16Fnc49678v38'
api_url = 'https://generativelanguage.googleapis.com'

# Initialize the pyttsx3 engine for text-to-speech
engine = pyttsx3.init()


def record_audio(duration=5, sample_rate=16000):
    """Records audio from the microphone for the given duration."""
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio, sample_rate
def save_audio_to_file(audio, sample_rate):
    """Saves the recorded audio to a WAV file."""
    temp_filename = tempfile.mktemp(suffix='.wav')
    write(temp_filename, sample_rate, audio)
    return temp_filename
def transcribe_audio(file_path):
    """Transcribes the audio file using OpenAI Whisper."""
    headers = {
        'Authorization': f'Bearer {api_key}',
    }

    for attempt in range(5):  # Retry up to 5 times
        with open(file_path, 'rb') as f:
            response = requests.post(api_url, headers=headers, files={'file': f}, data={'model': 'whisper-1'})

        if response.status_code == 200:
            result = response.json()
            return result.get('text', 'No transcription text found.')
        elif response.status_code == 429:  # Rate limit exceeded
            print("Quota exceeded, retrying in 30 seconds...")
            time.sleep(30)  # Wait before retrying
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    print("Failed to transcribe audio after multiple attempts.")
    return None

def speak(text):
    """Speaks the given text using pyttsx3."""
    try:
        engine.setProperty('rate', 150)  # Set speaking speed
        engine.setProperty('volume', 1.0)  # Set volume (0.0 to 1.0)

        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)  # Use the first available voice

        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error during speech synthesis: {e}")

def load_data(file_path):
    """Loads an Excel file containing sales, inventory, and purchasing data."""
    data = pd.read_excel(file_path)
    print("Data loaded successfully.")
    return data

def analyze_sales(data):
    """Analyzes the sales data and returns a summary of total sales per product."""
    sales_summary = data.groupby('Product')['Sales'].sum().sort_values(ascending=False)
    print("Sales analysis complete.")
    return sales_summary

def analyze_inventory(data):
    """Analyzes the inventory data and returns a summary of stock levels per product."""
    inventory_summary = data.groupby('Product')['Stock'].mean().sort_values(ascending=True)
    print("Inventory analysis complete.")
    return inventory_summary

def plot_sales(sales_summary):
    """Generates a bar chart for the top-selling products."""
    fig = px.bar(sales_summary, x=sales_summary.index, y=sales_summary.values,
                 labels={'x': 'Product', 'y': 'Total Sales'}, title="Top-Selling Products")
    fig.show()

def plot_inventory(inventory_summary):
    """Generates a bar chart for inventory stock levels per product."""
    fig = px.bar(inventory_summary, x=inventory_summary.index, y=inventory_summary.values,
                 labels={'x': 'Product', 'y': 'Stock Levels'}, title="Inventory Levels")
    fig.show()

def forecast_sales(data, product_name):
    """Forecast sales for a specific product using Prophet."""
    product_data = data[data['Product'] == product_name][['Date', 'Sales']]

    # Prepare data for Prophet
    product_data = product_data.rename(columns={'Date': 'ds', 'Sales': 'y'})

    # Initialize and fit the model
    model = Prophet()
    model.fit(product_data)

    # Make future predictions (e.g., 30 days into the future)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)
    plt.show()

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def visualize_predictions_vs_actuals(y_test, predictions):
    """Visualizes predictions versus actual stockout status."""
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

    # Count occurrences
    result_counts = results.groupby(['Actual', 'Predicted']).size().reset_index(name='Count')

    # Create bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Actual', y='Count', hue='Predicted', data=result_counts, palette='viridis')
    plt.title('Stockout Predictions vs. Actuals')
    plt.xlabel('Actual Stockout Status')
    plt.ylabel('Count')
    plt.legend(title='Predicted')
    plt.show()

def plot_confusion_matrix(y_test, predictions):
    """Plots a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stockout', 'Stockout'],
                yticklabels=['No Stockout', 'Stockout'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_classification_report(y_test, predictions):
    """Plots a bar chart of precision, recall, and F1-score from the classification report."""
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = ['precision', 'recall', 'f1-score']

    for metric in metrics:
        scores = {key: report[key][metric] for key in report if key not in ('accuracy', 'macro avg', 'weighted avg')}
        df_scores = pd.DataFrame(list(scores.items()), columns=['Class', metric])

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Class', y=metric, data=df_scores, palette='viridis')
        plt.title(f'{metric.capitalize()} by Class')
        plt.xlabel('Class')
        plt.ylabel(metric.capitalize())
        plt.show()

def predict_stockout(data):
    """Predict stockouts using a classification model."""
    data['Stockout'] = (data['Stock'] == 0).astype(int)

    # Features and target variable
    X = data[['Sales', 'Stock', 'Purchases']]
    y = data['Stockout']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict stockouts on the test set
    predictions = model.predict(X_test)

    # Print diagnostic information
    print("Stockout predictions (Test set):")
    print(predictions)

    # Visualize results
    visualize_predictions_vs_actuals(y_test, predictions)
    plot_confusion_matrix(y_test, predictions)
    plot_classification_report(y_test, predictions)

    print("Stockout predictions completed.")
    return predictions

def optimize_purchasing(data, product_name):
    """Optimize purchasing decisions based on forecasted sales and inventory levels."""
    product_data = data[data['Product'] == product_name]

    # Forecast future sales for the product
    forecast = forecast_sales(data, product_name)

    # Current inventory level
    current_stock = product_data['Stock'].iloc[-1]

    # Calculate optimal purchasing decision (simple heuristic)
    required_stock = forecast['yhat'].sum()  # Total forecasted sales
    suggested_purchase = max(0, required_stock - current_stock)  # Suggest purchase if needed

    print(f"Suggested Purchase for {product_name}: {suggested_purchase} units")

    # Plot the forecasted sales and current stock
    dates = forecast['ds']
    predicted_sales = forecast['yhat']

    fig = px.line(x=dates, y=predicted_sales, labels={'x': 'Date', 'y': 'Forecasted Sales'},
                  title=f'Forecasted Sales and Current Stock for {product_name}')

    # Add current stock as a horizontal line
    fig.add_scatter(x=dates, y=[current_stock] * len(dates), mode='lines', name='Current Stock',
                    line=dict(color='red', dash='dash'))

    fig.show()

    return suggested_purchase

def normalize_command(command):

    if command is None:
        return ""
    return command.lower().replace('-', ' ')

def interactive_assistant(file_path):

    data = load_data(file_path)

    while True:
        print("Waiting for voice command...")
        audio, sample_rate = record_audio()
        file_path = save_audio_to_file(audio, sample_rate)
        command = transcribe_audio(file_path)
        os.remove(file_path)  # Clean up the temporary file


        normalized_command = normalize_command(command)

        print(f"Normalized command: {normalized_command}")

        if 'sales analysis' in normalized_command:
            sales_summary = analyze_sales(data)
            plot_sales(sales_summary)
            response_text = "I have completed the sales analysis."

        elif 'inventory analysis' in normalized_command:
            inventory_summary = analyze_inventory(data)
            plot_inventory(inventory_summary)
            response_text = "Inventory analysis is done."

        elif 'forecast sales' in normalized_command:
            product_name = input("Enter product name for sales forecast: ")
            forecast = forecast_sales(data, product_name)
            response_text = f"Sales forecast for {product_name} completed."

        elif 'predict stockout' in normalized_command or 'stock out prediction' in normalized_command:
            predictions = predict_stockout(data)
            response_text = "Stockout predictions completed."

        elif 'optimize purchasing' in normalized_command:
            product_name = input("Enter product name for purchase optimization: ")
            optimize_purchasing(data, product_name)
            response_text = f"Purchase optimization for {product_name} completed."

        elif 'exit' in normalized_command:
            response_text = "Goodbye! Exiting the assistant."
            speak(response_text)
            break

        else:
            response_text = "Sorry, I couldn't understand the command."

        print(f"AI: {response_text}")
        speak(response_text)

if __name__ == "__main__":

    file_path = 'sales_file.xlsx'
    interactive_assistant(file_path)