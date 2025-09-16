# 🌎 Climate AI Product

An AI-powered climate literacy dashboard built with Streamlit.

## Features
- Real-time weather with OpenWeather API
- 5-day forecast visualization
- AI climate recommendations
- Mini climate chatbot (based on local climate_facts.txt)
- Clean dark professional UI

## Run
```bash
cd "C:\Users\Dell\OneDrive\Desktop\Climate_AI_Pro"
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py


## Data
- `data/climate_facts.txt` contains knowledge base for chatbot.
