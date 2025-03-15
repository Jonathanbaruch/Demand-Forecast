### Import Libraries

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
import statsmodels.api as sm
import streamlit as st


### Functions ###

### Generate Dummy Data

def generate_data(start_date, end_date, weather, base_seatings):
    data = []
    current_date = start_date
    time_slots = [17, 18, 19, 20, 21, 22, 23]  # Restaurant open from 17 to 23

    
    while current_date <= end_date:
        weekday = current_date.weekday() + 1  # Monday=1, Sunday=7
        event = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% chance of event

        # Get weather of current
        temp_row = weather.loc[weather["date"] == current_date, "temperature_2m_max"]
        rain_row = weather.loc[weather["date"] == current_date, "precipitation_sum"]

        # Extract single values or return 0 if missing or zero
        temperature = temp_row.values[0] if not temp_row.empty and temp_row.values[0] > 0 else 0
        rain = rain_row.values[0] if not rain_row.empty and rain_row.values[0] > 0 else 0

        
        temp_boost = 1.1 if temperature > 10 else 1  # Boost if warm weather
        temp_boost2 = 1.3 if temperature > 18 else 1
        rain_decrease = 0.8 if rain == 1 else 1  # Decrease if raining
        event_boost = 1.5 if event == 1 else 1  # Boost if an event is occurring
        
        # Create Seatings for each hour in the day
        for hour in time_slots:
            base_seatings = base_seatings  # Base seatings outside peak hours
            peak_factor = 1.5 if 18 <= hour <= 20 else 1  # Peak time boost
            peak_factor2 = 1.2 if 20 < hour <= 21 else 1  # Peak time boost
            weekend_boost = 1.25 if weekday in [5, 6] else 1  # Fridays and Saturdays get more seatings
            
            seatings = round(np.random.normal(base_seatings * peak_factor * peak_factor2 * weekend_boost * temp_boost * temp_boost2 * rain_decrease * event_boost, scale=1),0)
            
            data.append([current_date, hour, weekday, seatings, rain, temperature, event])
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(data, columns=["Date", "Hour", "Weekday", "Actual_Seatings", "Rain", "Temperature", "Event"])




### Get Weather

def weather_api(start_date):

    # Define start date (fixed) and calculate past_days dynamically
    start_date = start_date
    today = datetime.now()
    past_days = (today - start_date).days

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 55.6759,
        "longitude": 12.5655,
        "daily": ["temperature_2m_max", "precipitation_sum"],
        "timezone": "Europe/Berlin",
        "past_days": past_days
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation {response.Elevation()} m asl")
    # print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

                                # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(1).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}


    #round to 0 decimals
    daily_data["temperature_2m_max"] = np.round(daily_temperature_2m_max, 0)
    daily_data["precipitation_sum"] = np.round(daily_precipitation_sum, 0)

    daily_dataframe = pd.DataFrame(data = daily_data)

    ## Shift date 1 day forward (as the data is apparently for the day after). so the api apparently sets the day as the day before (i checked against the website)
    # Shift the Date column by +1 day
    daily_dataframe['date'] = daily_dataframe['date'] + pd.Timedelta(days=1)

    # Remove timezone from the datetime
    #daily_dataframe['date'] = daily_dataframe['date'].dt.date

    # Normalize the datetime and remove the timezone in one line
    daily_dataframe['date'] = daily_dataframe['date'].dt.normalize().dt.tz_localize(None)


    weather = daily_dataframe


    return weather




### Do Simple Machine Learning

def machine_learning(df, end_date):

    ### Machine Learning ###
    # Learn from data up until today
    # Forecast for the next 7 days (and simulate old forecasts)

    # Define the training data
    X = df[["Date", "Weekday", "Hour", "Rain", "Temperature", "Event"]]
    y = df["Actual_Seatings"]

    # X withwout date to predict on
    X_f = X.copy()
    X_f = X_f.drop(columns=["Date"])

    # Split into train and forecast based on start_date and end_date
    X_train = X[X["Date"] < end_date]
    y_train = y[X["Date"] < end_date]

    # Remove the date column
    X_train = X_train.drop(columns=["Date"])
    
    # One hot encode the weekday & Hour
    X_train = pd.get_dummies(X_train, columns=["Weekday", "Hour"], dtype=int)
    X_f = pd.get_dummies(X_f, columns=["Weekday", "Hour"], dtype=int)

    # Add a constant for the intercept
    X_train = sm.add_constant(X_train, has_constant='add')
    X_f = sm.add_constant(X_f, has_constant='add')

    # Train the model
    model = sm.OLS(y_train, X_train).fit()
    #print(model.summary2())

    # Predict the forecast
    y_forecast = model.predict(X_f)
    y_forecast = y_forecast.clip(0)  # Remove negative values

    # Add the forecast to the dataframe
    df["Forecast_Seatings"] = round(y_forecast,0)

    # Calculate residuals
    df["Residuals"] = df["Actual_Seatings"] - df["Forecast_Seatings"]

    return df




### Convert Seatings to Employees

def employees(df):
    ## Add employees based on Actual_Seatings ##

    # Remove Actual_Seatings from days in future
    df["Actual_Seatings"] = df["Actual_Seatings"].where(df["Date"] < datetime.now(), 0)

    ### Convert Actual_Seatings to Employees ###
    # Predicted employees:
    df['Forecasted_Employees'] = pd.cut(df['Forecast_Seatings'], bins=[0, 3, 8, 13, 17, 22, 40], labels=[1, 2, 3, 4, 5, 6])

    # Optimal employees:
    df['Optimal_Employees'] = pd.cut(df['Actual_Seatings'], bins=[0, 3, 8, 13, 17, 22, 40], labels=[1, 2, 3, 4, 5, 6])

    # Actual Employees Dummy:
    df['Actual_Employees'] = pd.cut(df['Actual_Seatings'] * 1.2, bins=[0, 3, 8, 13, 17, 22, 40], labels=[1, 2, 3, 4, 5, 6])

    return df




### Combine All Functions into one ###



def main(start_date, end_date, restaurants, base_seatings):
    # Get weather data
    weather = weather_api(start_date=start_date)

    # Ensure session_state has storage for restaurant DataFrames
    if "restaurant_data" not in st.session_state:
        st.session_state.restaurant_data = {}

    # Apply functions to each restaurant
    for restaurant in restaurants:
        # Generate dataframe
        df = generate_data(start_date, end_date, weather, base_seatings=base_seatings[restaurants.index(restaurant)])
        df = machine_learning(df, end_date)
        df = employees(df)
        
        # Store DataFrame in session state
        st.session_state.restaurant_data[restaurant] = df
        print(f"Dataset for {restaurant} stored in session_state")
