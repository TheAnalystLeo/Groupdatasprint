import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import streamlit as st


hdb = pd.read_csv('./datasets/hdb_streamlit.csv')


feature_cols = [
    'floor_area_sqm', 'exec_sold', '5room_sold', 'max_floor_lvl',
    'Hawker_Nearest_Distance',
    'zone_east', 'zone_north', 'zone_south', 'zone_west']

# Define X and y
X = hdb[feature_cols]
y = hdb['resale_price']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
# Build and fit the model
rf_model = RandomForestRegressor(random_state=123)
rf_model.fit(X_train, y_train)



# Streamlit UI
st.title(":house: HDB Resale Price Predictor")
st.write("Please fill in the values below:")
# User inputs
floor_area = st.number_input("Floor Area (sqm)", min_value=31, max_value=166, value=31, step=1)
exec_sold = st.number_input("Executive Flats Sold Nearby", min_value=0, max_value=132, value=0, step=1)
five_room_sold = st.number_input("5-Room Flats Sold Nearby", min_value=0, max_value=161, value=0, step=1)
max_floor_lvl = st.number_input("Max Floor Level", min_value=6, max_value=22, value=6, step=1)
hawker_dist = st.number_input("Distance to Nearest Hawker Centre (m)", min_value=9, max_value=3635, value=100, step=1)
# Dropdown for region
region = st.selectbox('Town Region', ['North', 'South', 'East', 'West'])
# Step 1: Create dummy variables for region
region_features = {
    'zone_north': 0,
    'zone_south': 0,
    'zone_east': 0,
    'zone_west': 0
}
region_key = f"zone_{region.lower()}"
region_features[region_key] = 1
# Step 2: Combine all features into the input list
input_list = [
    floor_area,
    exec_sold,
    five_room_sold,
    max_floor_lvl,
    hawker_dist,
    region_features['zone_east'],
    region_features['zone_north'],
    region_features['zone_south'],
    region_features['zone_west']
]
# Step 3: Convert to numpy array and predict
input_features = np.array([input_list])
predicted_price = rf_model.predict(input_features)[0]
# Display result
st.subheader(":chart_with_upwards_trend: Predicted Resale Price")
st.success(f"${predicted_price:,.0f}")