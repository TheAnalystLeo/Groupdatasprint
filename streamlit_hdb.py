import pandas as pd
import joblib
import streamlit as st

# Load the model
loaded_model = joblib.load(r"C:\Users\benwr\OneDrive\Desktop\Github_repo\Groupdatasprint\random_forest_model.pkl")  # Adjust the path as necessary

# Streamlit UI
st.title(":house: HDB Resale Price Predictor")
st.write("Please fill in the values below to predict the resale price:")

# User inputs
floor_area = st.number_input("Floor Area (sqm)", min_value=31, max_value=166, value=31, step=1)
exec_sold = st.number_input("Executive Flats Sold Nearby", min_value=0, max_value=132, value=0, step=1)
five_room_sold = st.number_input("5-Room Flats Sold Nearby", min_value=0, max_value=161, value=0, step=1)
max_floor_lvl = st.number_input("Max Floor Level", min_value=6, max_value=22, value=6, step=1)
hawker_dist = st.number_input("Distance to Nearest Hawker Centre (m)", min_value=9, max_value=3635, value=100, step=1)

# Dropdown for region selection
region = st.selectbox('Town Region', ['North', 'South', 'East', 'West'])

# Prepare dummy variables for the region
region_features = {
    'zone_north': 0,
    'zone_south': 0,
    'zone_east': 0,
    'zone_west': 0
}
if region == "North":
    region_features['zone_north'] = 1
elif region == "South":
    region_features['zone_south'] = 1
elif region == "East":
    region_features['zone_east'] = 1
elif region == "West":
    region_features['zone_west'] = 1
    
# Prepare input features as a DataFrame
input_features = {
    'floor_area_sqm': floor_area,
    'exec_sold': exec_sold,
    '5room_sold': five_room_sold,
    'max_floor_lvl': max_floor_lvl,
    'Hawker_Nearest_Distance': hawker_dist,
    'zone_north': region_features['zone_north'],
    'zone_south': region_features['zone_south'],
    'zone_east': region_features['zone_east'],
    'zone_west': region_features['zone_west']
}
input_df = pd.DataFrame(input_features, index=[0])  # Create DataFrame from input features
# Make prediction based on features when the button is pressed
if st.button("Predict"):
    predicted_price = loaded_model.predict(input_df)[0]
    st.subheader(":chart_with_upwards_trend: Predicted Resale Price")
    st.success(f"${predicted_price:,.2f}")