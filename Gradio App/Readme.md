---
title: British Airways Booking Prediction
emoji: 🛫
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🛫 British Airways Booking Prediction

This machine learning application predicts whether a customer will complete their booking based on various factors such as flight details, purchase timing, and additional services.

## 🚀 Features

- **Real-time Predictions**: Get instant booking completion predictions
- **User-friendly Interface**: Easy-to-use Gradio interface
- **Confidence Scoring**: See prediction confidence levels
- **Multiple Input Factors**: Considers sales channel, timing, flight details, and services

## 📊 Model Details

- **Algorithm**: CatBoost Classifier
- **Features**: 13 engineered features including purchase lead time, flight details, and service preferences
- **Preprocessing**: MinMax scaling for numerical features
- **Performance**: Optimized for booking prediction accuracy

## 🛠️ Usage

1. Select your sales channel (Internet/Mobile)
2. Choose flight day and time
3. Enter booking details (lead time, stay duration, flight duration)
4. Select additional services (baggage, seat, meals)
5. Click "Predict Booking Completion" to get results

## 🏗️ Technical Stack

- **Frontend**: Gradio
- **ML Framework**: CatBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Deployment**: Hugging Face Spaces

## 📈 Model Features

The model considers the following factors:
- Sales channel preference
- Purchase lead time
- Length of stay
- Flight timing (hour and day)
- Additional service preferences
- Derived features (weekend flights, time periods, ratios)

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## 📝 License

This project is licensed under the MIT License.