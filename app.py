import streamlit as st
import requests
import os
import plotly.graph_objects as go
import sys  # For debug

# -----------------------------
# Set Hugging Face cache folder (Streamlit Cloud friendly)
# -----------------------------
os.environ['HF_HOME'] = '/tmp/huggingface_cache'

# Optional: prevent repeated downloads
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# Debug: show Python interpreter path
st.write(f"Running with Python interpreter: {sys.executable}")

# -----------------------------
# Load API key from Streamlit Secrets
# -----------------------------
API_KEY = st.secrets["API_KEY"]
BASE_URL = "http://api.openweathermap.org/data/2.5/"

# -----------------------------
# Load models (small / cloud optimized)
# -----------------------------
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Summarization model (~250MB)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Sentence embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load climate corpus
# -----------------------------
if not os.path.exists("climate.txt"):
    st.error("climate.txt not found! Please add it to the repo root.")
    st.stop()

with open("climate.txt", "r", encoding="utf-8") as f:
    climate_corpus = f.read().splitlines()
corpus_embeddings = embedder.encode(climate_corpus, convert_to_tensor=True)

# -----------------------------
# Weather & Forecast Functions
# -----------------------------
def get_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def get_forecast(city):
    url = f"{BASE_URL}forecast?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def ai_recommendation(condition):
    condition = condition.lower()
    if "rain" in condition:
        return "Carry an umbrella and avoid waterlogging areas."
    elif "clear" in condition:
        return "Perfect day for solar charging and outdoor activities."
    elif "cloud" in condition:
        return "Mild weather, but carry a light jacket."
    elif "snow" in condition:
        return "Wear warm clothes and avoid slippery roads."
    elif "storm" in condition:
        return "Stay indoors and keep devices charged."
    else:
        return "Stay safe and follow eco-friendly practices."

# -----------------------------
# AI Functions
# -----------------------------
def chatbot_response(query):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    best_idx = scores.argmax().item()
    context = climate_corpus[best_idx]
    result = qa_pipeline(question=query, context=context)
    return result["answer"]

def generate_summary(text, max_len=150):
    return summarizer(text, max_length=max_len, min_length=50, do_sample=False)[0]["summary_text"]

def plot_forecast_chart(forecast_data):
    dates, temps = [], []
    for item in forecast_data["list"][:5]:
        dates.append(item["dt_txt"])
        temps.append(item["main"]["temp"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=temps, mode="lines+markers", name="Temperature"))
    fig.update_layout(
        title="5-Day Weather Forecast",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        template="plotly_dark"
    )
    return fig

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="AI Climate & Weather Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        body {background-color: #0e1117; color: white;}
        .big-temp {font-size: 48px; font-weight: bold; color: #00c3ff;}
        </style>
        """, unsafe_allow_html=True)

    st.title("🌍 AI-Powered Climate & Weather Dashboard")

    city = st.text_input("Enter City:", "Mumbai")
    if city:
        weather = get_weather(city)
        forecast = get_forecast(city)

        if weather:
            temp = weather["main"]["temp"]
            condition = weather["weather"][0]["description"]
            st.markdown(f"<p class='big-temp'>{temp}°C</p>", unsafe_allow_html=True)
            st.write(f"**Condition:** {condition.title()}")
            st.write(f"**AI Suggestion:** {ai_recommendation(condition)}")

        if forecast:
            st.plotly_chart(plot_forecast_chart(forecast), use_container_width=True)

    st.subheader("🧠 Climate Chatbot")
    user_q = st.text_input("Ask about Climate Change:")
    if user_q:
        answer = chatbot_response(user_q)
        st.success(f"🤖 {answer}")

    st.subheader("📖 Climate Summary Generator")
    user_text = st.text_area("Paste Climate-related text:")
    if st.button("Generate Summary"):
        if user_text.strip():
            summary = generate_summary(user_text)
            st.info(summary)

if __name__ == "__main__":
    main()


