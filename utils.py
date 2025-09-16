
import os
import requests
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "fe141fd3058ed253c214f5dff7fe69a5"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

def get_weather(city:str):
    try:
        r = requests.get(BASE_URL, params={"q": city, "appid": API_KEY, "units": "metric"}, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def get_forecast(city:str):
    try:
        r = requests.get(FORECAST_URL, params={"q": city, "appid": API_KEY, "units": "metric"}, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        rows = []
        for item in data.get("list", []):
            dt = datetime.fromtimestamp(item["dt"])
            day = dt.date()
            temp = item["main"]["temp"]
            rows.append({"date": day, "temp": temp})
        if not rows:
            return None
        df = pd.DataFrame(rows).groupby("date", as_index=False).mean().head(5)
        return df
    except Exception:
        return None

def ai_recommendations(weather:dict):
    tips = ["Use public transport or carpool to cut emissions", "Switch to LED bulbs and turn off idle devices", "Carry a reusable bottle and bag", "Reduce food waste and prefer local produce", "Consider a meat-light week to lower footprint"]
    if not weather:
        return tips
    if weather['main']['temp'] > 30:
        tips.append("Stay hydrated and avoid peak heat hours")
    if weather['main']['humidity'] > 70:
        tips.append("Use energy-efficient cooling methods")
    return tips

def load_facts(path="data/climate_facts.txt"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

_model = None
def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def climate_chatbot(query, path="data/climate_facts.txt"):
    facts = load_facts(path)
    if not facts:
        return "No climate knowledge base found."
    model = _get_model()
    q_emb = model.encode([query])
    f_emb = model.encode(facts)
    sims = cosine_similarity(q_emb, f_emb)[0]
    idx = sims.argmax()
    return facts[idx]
