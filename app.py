import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("Flight.h5")
pickle_in = open("scaler.pkl", "rb")  ## rb = READ BYTE
scaler = pickle.load(pickle_in)

# Function to preprocess the input features
def preprocess_input(inp):
    return scaler.transform([inp])

# Function to make the flight amount prediction
def predict_flight_amount(inp):
    # Make the prediction using the loaded model
    prediction = model.predict(inp)
    return prediction[0]

# Dictionary of helicopter details
helicopter_details = {
    "The R66 Robinson helicopter": {
        "Capacity": 6,
        "Fuel Efficiency (L per Hour)": 87,
        "Max Speed (KM/h)": 250,
        "Hold carries up to (pounds)": 300,
    },
}

# Streamlit app
def main():
    st.title("Indian Private Helicopter Fare")

    # Display helicopter details
    st.subheader("Helicopter Details: The R66 Robinson Helicopter")
    for helicopter_name, details in helicopter_details.items():
        st.write(f"Helicopter Name: {helicopter_name}")
        st.write(f"Capacity: {details['Capacity']} passengers")
        st.write(f"Fuel Efficiency: {details['Fuel Efficiency (L per Hour)']} L per Hour")
        st.write(f"Max Speed: {details['Max Speed (KM/h)']} KM/h")
        st.write(f"Hold Capacity: {details['Hold carries up to (pounds)']} pounds")
        st.write("---")

    # Display helicopter image
    image = "image.jpg"  # Path to your helicopter image
    st.image(image, caption="The R66 Robinson Helicopter", use_column_width=True)

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Input features
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=5)
    h_distance = st.number_input("Haversine Distance (KM)", min_value=0.1, max_value=200.0, step=0.1)
    year = st.number_input("Year", min_value=2010, max_value=2030)
    month = st.number_input("Month", min_value=1, max_value=12)
    date = st.number_input("Date", min_value=1, max_value=31)
    day_of_week = st.selectbox("Day of Week", days)
    day_of_week = days.index(day_of_week)
    hour = st.number_input("Hour", min_value=0, max_value=23)

    # Preprocess input
    inp = [passenger_count, h_distance, year, month, date, day_of_week, hour]
    input_data = preprocess_input(inp)

    # Make prediction
    prediction = predict_flight_amount(input_data)

    # Display the prediction
    st.subheader("Flight Amount Prediction")
    st.write(f"The predicted flight amount is: {prediction[0]} $")

if __name__ == '__main__':
    main()
