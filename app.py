# app.py

# ==== Imports ====
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


# ==== Load Trained Components ====
CBC = joblib.load('catboost_model.pkl')
sc = joblib.load('scaler.pkl')


# ==== Helper Functions ====
def hour_bin(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'night'


def predict_booking(sales_channel, purchase_lead, length_of_stay, flight_hour, 
                   flight_day, wants_extra_baggage, wants_preferred_seat, 
                   wants_in_flight_meals, flight_duration):
    try:
        stay_per_passenger = length_of_stay / 2  # Approximation
        lead_time_ratio = purchase_lead / (length_of_stay + 1)
        is_weekend_flight = 1 if flight_day in ['Saturday', 'Sunday'] else 0
        flight_period = hour_bin(flight_hour)

        # Encodings (same used during training)
        sales_channel_map = {'Internet': 0, 'Mobile': 1}
        flight_day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                         'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        flight_period_map = {'morning': 0, 'afternoon': 1, 'night': 2}

        sales_channel_encoded = sales_channel_map.get(sales_channel, 0)
        flight_day_encoded = flight_day_map.get(flight_day, 0)
        flight_period_encoded = flight_period_map.get(flight_period, 0)

        features = np.array([[
            sales_channel_encoded,
            purchase_lead,
            length_of_stay,
            flight_hour,
            flight_day_encoded,
            int(wants_extra_baggage),
            int(wants_preferred_seat),
            int(wants_in_flight_meals),
            flight_duration,
            stay_per_passenger,
            lead_time_ratio,
            is_weekend_flight,
            flight_period_encoded
        ]])

        features_scaled = sc.transform(features)

        prediction = CBC.predict(features_scaled)[0]
        probability = CBC.predict_proba(features_scaled)[0]

        result = "✅ Will Complete Booking" if prediction == 1 else "❌ Will Not Complete Booking"
        confidence = f"Confidence: {max(probability):.2%}"

        return f"{result}\n{confidence}"

    except Exception as e:
        return f"❗ Error in prediction: {str(e)}"


# ==== Gradio Interface ====
def create_gradio_app():
    with gr.Blocks(title="British Airways Booking Prediction", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🛫 British Airways Booking Prediction
        Predict whether a customer will complete their booking based on various factors.
        """)

        with gr.Row():
            with gr.Column():
                sales_channel = gr.Dropdown(["Internet", "Mobile"], label="Sales Channel", value="Internet")
                flight_day = gr.Dropdown(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], label="Flight Day", value="Monday")
                flight_hour = gr.Slider(0, 23, step=1, label="Flight Hour (24-hr)", value=12)
                flight_duration = gr.Number(label="Flight Duration (hours)", value=2.5, minimum=0.5, maximum=15)
            with gr.Column():
                purchase_lead = gr.Number(label="Purchase Lead Time (days)", value=30, minimum=0, maximum=365)
                length_of_stay = gr.Number(label="Length of Stay (days)", value=7, minimum=1, maximum=30)
                wants_extra_baggage = gr.Checkbox(label="Wants Extra Baggage", value=False)
                wants_preferred_seat = gr.Checkbox(label="Wants Preferred Seat", value=False)
                wants_in_flight_meals = gr.Checkbox(label="Wants In-Flight Meals", value=False)

        predict_btn = gr.Button("🚀 Predict Booking Completion", variant="primary", size="lg")
        output = gr.Textbox(label="Prediction Result", lines=3)

        predict_btn.click(
            fn=predict_booking,
            inputs=[
                sales_channel, purchase_lead, length_of_stay, flight_hour,
                flight_day, wants_extra_baggage, wants_preferred_seat,
                wants_in_flight_meals, flight_duration
            ],
            outputs=output
        )

        gr.Markdown("""
        ### Instructions:
        1. Fill in the flight and booking details
        2. Click the button to predict booking completion
        3. You'll see a prediction and confidence score
        """)

    return app


# ==== Launch ====
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()
