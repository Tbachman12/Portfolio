# 🏀 NBA Player Points Predictor

A machine learning web app that predicts how many points an NBA player 
will score in their next game based on their 2025-26 season stats.

## 📊 Model Performance
- **Mean Absolute Error:** 1.85 points
- **R² Score:** 0.95

## 🛠️ Built With
- Python, Scikit-learn, NBA API, Streamlit, Pandas

## 🚀 How to Run
1. pip install -r requirements.txt
2. python get_data.py
3. streamlit run app.py

## 💡 How It Works
- Pulls live 2025-26 NBA game logs via the NBA API
- Averages a player's last 5 games as prediction input
- Random Forest Regressor outputs predicted points




