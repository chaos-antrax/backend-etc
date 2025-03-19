from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import requests
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load('saved Models/logistic_model.pkl')
encoder = joblib.load('saved Models/encoder.pkl')
scaler = joblib.load('saved Models/scaler.pkl')
xgb_model = joblib.load('saved Models/xgb_model.pkl')
lstm_model = load_model('saved Models/best_lstm_fuel_price.h5')



cred = credentials.Certificate('firebase-growth.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

categorical_columns = ['Crop type', 'Soil Type']
numerical_columns = ['PH Value', 'Potassium (ppm)', 'Phosphorus (ppm)', 
                     'Sunlight Hours', 'Temperature (°C)', 'Humidity (%)']


data_fuel = pd.read_csv('Data Set/Fuel_Price_Prediction_Data_Set.csv') 
data_fuel['Date'] = pd.to_datetime(data_fuel['Date'])
data_fuel.set_index("Date", inplace=True)
data_fuel = data_fuel[["Market Diesel price", "Market petrol price"]]
data_fuel["Month_sin"] = np.sin(2 * np.pi * data_fuel.index.month / 12)
data_fuel["Month_cos"] = np.cos(2 * np.pi * data_fuel.index.month / 12)
data_fuel["Day_sin"] = np.sin(2 * np.pi * data_fuel.index.day / 31)
data_fuel["Day_cos"] = np.cos(2 * np.pi * data_fuel.index.day / 31)

df_scaled_data = data_fuel.drop(columns=["Market Diesel price", "Market petrol price"]) 

data_fuel = data_fuel.sort_values(by="Date")

scalerFuel = MinMaxScaler(feature_range=(0, 1))
df_scaled = scalerFuel.fit_transform(data_fuel)

sequence_length = 60 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['PUT'])
def predict():
    try:
        # Get the input JSON data
        input_data = request.get_json()

        # Convert input to a DataFrame
        sample_data = pd.DataFrame([input_data])

        # Preprocess the data
        encoded_sample = encoder.transform(sample_data[categorical_columns])
        encoded_sample_df = pd.DataFrame(encoded_sample, columns=encoder.get_feature_names_out(categorical_columns))
        
        numerical_sample = sample_data[numerical_columns]
        numerical_sample_scaled = scaler.transform(numerical_sample)
        numerical_sample_df = pd.DataFrame(numerical_sample_scaled, columns=numerical_columns)

        # Combine encoded categorical and scaled numerical data
        final_sample = pd.concat([encoded_sample_df, numerical_sample_df], axis=1)

        # Predict class and probabilities
        predicted_class = model.predict(final_sample)[0]
        predicted_probabilities = model.predict_proba(final_sample)[0]

        # Create response
        response = {
            'predicted_class': 'Growth' if predicted_class == 1 else 'No Growth',
            'probability_growth': round(predicted_probabilities[1], 2),
            'probability_no_growth': round(predicted_probabilities[0], 2)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})
    
def get_weather_data(latitude, longitude, date):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max,temperature_2m_min,sunrise,sunset,relative_humidity_2m_max&timezone=Asia/Colombo&forecast_days=7"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Find the index of the requested date in the response
        try:
            index = data["daily"]["time"].index(date)
        except ValueError:
            return {"error": "Date not found in the forecast range"}

        # Extract required weather data for the given date
        max_temp = data["daily"]["temperature_2m_max"][index]
        min_temp = data["daily"]["temperature_2m_min"][index]
        humidity = data["daily"]["relative_humidity_2m_max"][index]
        sunrise = data["daily"]["sunrise"][index]
        sunset = data["daily"]["sunset"][index]

        # Calculate Temperature Difference
        temp_difference = max_temp

        # Calculate Sunlight Hours
        sunrise_time = datetime.strptime(sunrise, "%Y-%m-%dT%H:%M")
        sunset_time = datetime.strptime(sunset, "%Y-%m-%dT%H:%M")
        sunlight_hours = (sunset_time - sunrise_time).total_seconds() / 3600  # Convert to hours

        return {
            "humidity": humidity,
            "temperature_difference": round(temp_difference, 2),
            "sunlight_hours": round(sunlight_hours, 2)
        }
    else:
        return {"error": "Failed to retrieve data"}
# 📌 **Endpoint to Fetch Weather Data using Query Parameters**
@app.route('/get_weather_data', methods=['GET'])
def fetch_weather():
    try:
        # Extract parameters from the URL query
        latitude = request.args.get("latitude", default=6.9271, type=float)  # Default to Colombo
        longitude = request.args.get("longitude", default=79.8612, type=float)
        date = request.args.get("date")  # Expected format: YYYY-MM-DD

        if not date:
            return jsonify({"error": "Please provide a valid date (YYYY-MM-DD)"}), 400

        # Fetch Weather Data for Given Date
        weather_data = get_weather_data(latitude, longitude, date)

        return jsonify({"weather_data": weather_data})

    except Exception as e:
        return jsonify({"error": str(e)})
    
# Example local prediction
test_sample = pd.DataFrame({
    "Crop type": ["Capsicum"],
    "PH Value": [6.5],
    "Potassium (ppm)": [250],
    "Phosphorus (ppm)": [60],
    "Soil Type": ["Loamy Soil"],
    "Sunlight Hours": [8],
    "Temperature (°C)": [25],
    "Humidity (%)": [70]
})

# Preprocessing steps
encoded_sample = encoder.transform(test_sample[categorical_columns])
encoded_sample_df = pd.DataFrame(encoded_sample, columns=encoder.get_feature_names_out(categorical_columns))

numerical_sample = test_sample[numerical_columns]
numerical_sample_scaled = scaler.transform(numerical_sample)
numerical_sample_df = pd.DataFrame(numerical_sample_scaled, columns=numerical_columns)

final_sample = pd.concat([encoded_sample_df, numerical_sample_df], axis=1)

# Prediction
predicted_class = model.predict(final_sample)[0]
predicted_probabilities = model.predict_proba(final_sample)[0]

print("Predicted Class:", 'Growth' if predicted_class == 1 else 'No Growth')
print("Predicted Probabilities:", predicted_probabilities)


@app.route('/store_data', methods=['POST'])
def store_data():
    try:
        # Get the input data from the request
        input_data = request.get_json()

        # Here, you could process your prediction and weather data
        prediction_data = input_data['prediction_data']  # Data from /predict endpoint
        weather_data = input_data['weather_data']  # Data from /get_weather_data endpoint

        # Create a unique document ID based on timestamp
        document_id = str(datetime.utcnow().timestamp())

        # Combine prediction and weather data
        data_to_store = {
            'prediction': prediction_data,
            'weather': weather_data,
            'timestamp': datetime.utcnow().isoformat()  # Save timestamp when data is stored
        }

        # Store data in Firestore database
        db.collection('crop_growth_data').document(document_id).set(data_to_store)

        return jsonify({"status": "success", "message": "Data stored successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        input_data = request.get_json()

        predicted_price = predict_vegetable_price(input_data)
        return jsonify({'predicted_price': float(predicted_price)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Helper function for vegetable price prediction
def preprocess_input(input_data):
    input_data['Date'] = pd.to_datetime(input_data['Date'])
    input_data['Year'] = input_data['Date'].dt.year
    input_data['Month'] = input_data['Date'].dt.month
    input_data['Day'] = input_data['Date'].dt.day
    input_data['Weekday'] = input_data['Date'].dt.weekday
    input_data['DayOfYear'] = input_data['Date'].dt.dayofyear

    input_data.drop(columns=['Date'], inplace=True)

    categorical_features = ['Market', 'Type', 'Vegetable']
    training_features = xgb_model.get_booster().feature_names
    categories = [
        input_data['Market'].unique(), 
        input_data['Type'].unique(),
        input_data['Vegetable'].unique()
    ]
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=categories)  
    encoded_data = encoder.fit_transform(input_data[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names)
    input_data = input_data.drop(columns=categorical_features)
    processed_input = pd.concat([input_data, encoded_df], axis=1)
    processed_input = processed_input.reindex(columns=training_features, fill_value=0)
    processed_input = processed_input.astype(float)
    
    return processed_input

def predict_vegetable_price(input_data):
    input_df = pd.DataFrame([input_data])
    processed_input = preprocess_input(input_df)
    prediction = xgb_model.predict(processed_input)
    
    return prediction[0]

@app.route('/predict_fuel_price', methods=['POST'])
def predict_fuel_price():
    try:
        input_data = request.get_json()
        date_input = input_data['Date']

        # Process the input date for cyclical encoding (month, day)
        date_obj = pd.to_datetime(date_input)
        month_sin = np.sin(2 * np.pi * date_obj.month / 12)
        month_cos = np.cos(2 * np.pi * date_obj.month / 12)
        day_sin = np.sin(2 * np.pi * date_obj.day / 31)
        day_cos = np.cos(2 * np.pi * date_obj.day / 31)

        # Extract the last sequence of 60 data points from the scaled data
        last_sequence = df_scaled[-sequence_length:]

        placeholder_fuel_prices = np.zeros((1, 2))
        
        future_features = np.array([[month_sin, month_cos, day_sin, day_cos]])
        future_row = np.hstack([placeholder_fuel_prices, future_features])
        
        last_sequence = np.vstack([last_sequence, future_row])[-sequence_length:]
        input_sequence = np.array(last_sequence).reshape(1, sequence_length, df_scaled.shape[1])
        predicted_scaled_fuel = lstm_model.predict(input_sequence)
        predicted_original = scaler.inverse_transform(predicted_scaled_fuel)
        
        predicted_diesel = round(float(predicted_original[0, 0]), 2)  # Convert to Python float
        predicted_petrol = round(float(predicted_original[0, 1]), 2)
        
        return json.dumps({
        "date": date_input,
        "predicted_diesel_price": predicted_diesel,
        "predicted_petrol_price": predicted_petrol
    }, indent=4)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/get_all_data', methods=['GET'])
def get_all_data():
    try:
        # Fetch all documents from the 'crop_growth_data' collection
        docs = db.collection('crop_growth_data').stream()
        
        # Prepare the data to be returned
        all_data = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id  # Optionally add document ID for reference
            all_data.append(data)

        # Return all the documents as a JSON array
        return jsonify(all_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
 

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
