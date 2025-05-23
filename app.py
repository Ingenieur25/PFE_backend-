from flask import Flask, request, jsonify
from flask_cors import CORS  
import joblib
import pandas as pd

# Load the saved models, scalers, and polynomial transformers
gdp_model = joblib.load('linear_regression_model_degree_3.pkl')
gdp_scaler = joblib.load('scaler_degree_3.pkl')
gdp_poly = joblib.load('poly_degree_3.pkl')

life_expectancy_model = joblib.load('linear_regression_model_life_expectancy_year_only.pkl')
life_expectancy_scaler = joblib.load('scaler_life_expectancy_year_only.pkl')
life_expectancy_poly = joblib.load('poly_life_expectancy_year_only.pkl')

population_model = joblib.load('linear_regression_model_population_year_only.pkl')
population_scaler = joblib.load('scaler_population_year_only.pkl')
population_poly = joblib.load('poly_population_year_only.pkl')

# Initialize Flask app
app = Flask(__name__)

CORS(app, origins=[
    "https://pfe-frontend-bdaa.vercel.app",
    "http://localhost:3000"
])


# Route for predicting GDP
@app.route('/predict/gdp', methods=['GET'])
def predict_gdp():
    try:
        year = int(request.args.get('year'))
        future_data = pd.DataFrame([[year]], columns=['Year'])
        future_data_poly = gdp_poly.transform(future_data)
        future_data_scaled = gdp_scaler.transform(future_data_poly)
        future_gdp_prediction = gdp_model.predict(future_data_scaled)
        return jsonify({'predicted_gdp': float(future_gdp_prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Route for predicting Life Expectancy
@app.route('/predict/life_expectancy', methods=['GET'])
def predict_life_expectancy():
    try:
        year = int(request.args.get('year'))
        future_data = pd.DataFrame([[year]], columns=['Year'])
        future_data_poly = life_expectancy_poly.transform(future_data)
        future_data_scaled = life_expectancy_scaler.transform(future_data_poly)
        future_life_expectancy_prediction = life_expectancy_model.predict(future_data_scaled)
        return jsonify({'predicted_life_expectancy': float(future_life_expectancy_prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Route for predicting Population
@app.route('/predict/population', methods=['GET'])
def predict_population():
    try:
        year = int(request.args.get('year'))
        future_data = pd.DataFrame([[year]], columns=['Year'])
        future_data_poly = population_poly.transform(future_data)
        future_data_scaled = population_scaler.transform(future_data_poly)
        future_population_prediction = population_model.predict(future_data_scaled)
        return jsonify({'predicted_population': float(future_population_prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
