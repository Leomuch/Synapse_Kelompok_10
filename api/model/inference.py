import numpy as np
import pandas as pd
import joblib
import requests
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


class Model:
    def __init__(self, model_path):
        if model_path.endswith(".joblib"):
            self.model = joblib.load(model_path)
            self.model_type = "sklearn"
        else:
            raise ValueError(
                f"Model format '{model_path.split('.')[-1]}' not supported. Please use '.joblib'."
            )

    def data_pipeline(
        self, numerical_features=None, categorical_features=None, fit_data=None):
        if self.model_type != "sklearn":
            raise ValueError("Data pipeline is only supported for scikit-learn models.")

        # Define preprocessing for numerical features
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", MinMaxScaler()),
            ]
        )

        # Define preprocessing for categorical features
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder()),
            ]
        )

        # Combine preprocessors in a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="passthrough",
        )

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", self.model)])
        # Lakukan fit jika fit_data tersedia
        if fit_data is not None:
            pipeline.fit(fit_data)
        return pipeline

    def predict_from_data(
        self, data, numerical_features=None, categorical_features=None
    ):
        if self.model_type == "sklearn":
            if isinstance(data, (list, np.ndarray)):
                data = pd.DataFrame([data])
            elif not isinstance(data, pd.DataFrame):
                raise ValueError(
                    "Data format not supported for sklearn model. Use list, NumPy array, or DataFrame."
                )

            pipeline = self.data_pipeline(numerical_features, categorical_features)
            prediction = self.model.predict(data)

            prediction = (
                "drizzle"
                if prediction == 0
                else (
                    "fog"
                    if prediction == 1
                    else (
                        "rain"
                        if prediction == 2
                        else "snow" if prediction == 3 else "sun"
                    )
                )
            )
            return prediction

        else:
            raise ValueError("Model type not supported.")

    @staticmethod
    def from_path(model_path):
        return Model(model_path)

class WeatherAPI:
    def _init_(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.weatherapi.com/v1/forecast.json"

    def get_weather_data(self, city, days=8):
        url = f"{self.base_url}?key={self.api_key}&q={city}&days={days}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status code {response.status_code}")

    def extract_relevant_data(self, city_data):
        forecast_data = [
            {
                'maxtemp_c': day['day']['maxtemp_c'],
                'mintemp_c': day['day']['mintemp_c'],
                'maxwind_kph': day['day']['wind'],
                'totalprecip_mm': day['day']['totalprecip_mm']
            }
            for day in city_data['forecast']['forecastday']
        ]

        return pd.DataFrame(forecast_data)

def predict_weather_condition(forecast_df, model):
    numerical_features = ['maxtemp_c', 'mintemp_c', 'wind_kph', 'precip_mm']
    conditions = []
    for _, row in forecast_df.iterrows():
        # Prediksi kondisi cuaca menggunakan model
        data = row[numerical_features].values.reshape(1, -1)
        condition = model.predict_from_data(data, numerical_features, [])
        conditions.append(condition)
    
    forecast_df['condition'] = conditions
    return forecast_df



if __name__ == "_main_":
    # Load the trained model
    model_path = Path(_file_).parent / "garden" / "model.joblib"
    model = Model(model_path)  # Gunakan Model class

    # API Key and City
    api_key = "26f9fcacc03c434b99610156240212"
    cities = ["Samarinda", "Bontang", "Sangatta", "Berau", "Balikpapan", "Tenggarong", "Muara Badak", "Muara Wahau", "Sangkulirang", "Penajam"]
    # Fetch weather data and predict conditions for each city
    weather_api = WeatherAPI(api_key)
    
    for city in cities:
        weather_data = weather_api.get_weather_data(city)
        forecast_df = weather_api.extract_relevant_data(weather_data)
        forecast_with_condition = predict_weather_condition(forecast_df, model)

        # Display the forecast with predicted condition for each city
        print(f"Forecast for {city}:")
        print(forecast_with_condition)
        print("\n")