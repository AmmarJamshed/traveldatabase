#!/usr/bin/env python
# coding: utf-8

# In[1]:
st.set_page_config(page_title="SQL Travel Archetype", layout="wide")

import streamlit as st
import pandas as pd
import numpy as np
import random
import mysql.connector
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# --------------------------
# MySQL Config
# --------------------------
def get_connection():
    return mysql.connector.connect(
        host="localhost",  # or your DB host
        user="root",       # your username
        password="Ammarjamshed123",  # your password
        database="travel_db"
    )

# --------------------------
# Static Data
# --------------------------
badges = [
    "Inner + Flow", "Nature + Stillness", "Culture + Connection", "Community & Local First",
    "Food + Soul", "Wonder + Mystery", "Solo Traveler", "Off the Path", "Refinement + Aesthetics",
    "Luxury/Refined", "Budget-Friendly", "Adrenaline + Wild", "Urban + Discovery", "Slow & Soulful",
    "Heritage + History", "Holistic Ethical Travel", "Craft + Creation", "Journal to Self"
]

archetypes = [
    "Mindful Seeker", "Curious Connector", "Independent Explorer", "Earth Lover", "Elegant Voyager",
    "Cultural Alchemist", "Trailblazing Energizer", "Heartful Healer", "Radiant Nomad",
    "Structured Nomad", "Wild Mystic", "Offbeat Nomad", "Sensory Wanderer", "Inner Voyager",
    "Time Traveler", "Sacred Pilgrim", "Urban Soulwalker"
]

recommendations = {
    "Mindful Seeker": ["Bali", "Kyoto", "Kerala"],
    "Curious Connector": ["Lisbon", "Istanbul", "Buenos Aires"],
    "Independent Explorer": ["New Zealand", "Iceland", "Scotland"],
    "Earth Lover": ["Costa Rica", "Norwegian Fjords", "Patagonia"],
    "Elegant Voyager": ["Paris", "Vienna", "Florence"],
    "Cultural Alchemist": ["Marrakech", "Lahore", "Hanoi"],
    "Trailblazing Energizer": ["Peru", "South Africa", "Arizona"],
    "Heartful Healer": ["Sedona", "Rishikesh", "Ubud"],
    "Radiant Nomad": ["Thailand", "Mexico", "Portugal"],
    "Structured Nomad": ["Germany", "Singapore", "Canada"],
    "Wild Mystic": ["Amazon", "Tibet", "Madagascar"],
    "Offbeat Nomad": ["Tbilisi", "Uzbekistan", "Bhutan"],
    "Sensory Wanderer": ["Italy", "Morocco", "Thailand"],
    "Inner Voyager": ["Nepal", "Sri Lanka", "Greece"],
    "Time Traveler": ["Rome", "Cairo", "Athens"],
    "Sacred Pilgrim": ["Mecca", "Varanasi", "Jerusalem"],
    "Urban Soulwalker": ["New York", "Berlin", "Tokyo"]
}

# --------------------------
# Generate + Train Model
# --------------------------
def generate_user():
    return random.sample(badges, random.randint(3, 7))

@st.cache_data
def load_data():
    df = pd.DataFrame({
        "badges": [generate_user() for _ in range(200)],
        "archetype": [random.choice(archetypes) for _ in range(200)]
    })
    return df

df = load_data()
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["badges"])
y = df["archetype"]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3)
}

for name in models:
    models[name].fit(X, y)

# --------------------------
# Streamlit UI
# --------------------------
st.title("🌍 SQL-Powered Erranza Travel Archetype Engine")

model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))

view_user = st.sidebar.checkbox("🔎 View previous user recommendations")

if view_user:
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, username FROM user_recommendations ORDER BY created_at DESC LIMIT 50")
        users = cursor.fetchall()
        user_map = {f"{u['username']} (ID {u['id']})": u["id"] for u in users}
        if user_map:
            selected_user = st.sidebar.selectbox("Select user", list(user_map.keys()))
            if selected_user:
                user_id = user_map[selected_user]
                cursor.execute("SELECT * FROM user_recommendations WHERE id = %s", (user_id,))
                record = cursor.fetchone()
                st.write("### 🔁 Previously Stored Result:")
                st.write(f"**Username:** {record['username']}")
                st.write(f"**Badges:** {record['selected_badges']}")
                st.write(f"**Archetype:** {record['predicted_archetype']}")
                st.write(f"**Places:** {record['recommended_places']}")
        else:
            st.info("No past records found.")
    st.stop()

# Input Section
st.subheader("🔍 Enter Your Travel Preferences")
username = st.text_input("Enter your name:")
selected = st.multiselect("Select badges:", badges)

if st.button("🔮 Predict"):
    if not username or not selected:
        st.error("Name and badges are required.")
    else:
        input_vec = mlb.transform([selected])
        model = models[model_choice]
        pred = model.predict(input_vec)[0]
        recs = recommendations.get(pred, [])
        st.success(f"🎯 Your Travel Archetype: **{pred}**")
        st.write("### ✈️ Recommended Places")
        for r in recs:
            st.markdown(f"- {r}")

        # Save to MySQL
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_recommendations (username, selected_badges, predicted_archetype, recommended_places)
                    VALUES (%s, %s, %s, %s)
                """, (
                    username,
                    ",".join(selected),
                    pred,
                    ",".join(recs)
                ))
                conn.commit()
                st.success("✅ Saved to database.")
        except Exception as e:
            st.error(f"Failed to save to database: {e}")

