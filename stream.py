import pandas as pd
import streamlit as st
import plotly.express as px

st.title("Steps vs Pulse Rate")

data = pd.read_excel(r"C:\Users\KN\fitpulse_health_data.xlsx")

fig = px.scatter(
    data,
    x="steps",
    y="pulse_rate",
    title="Steps vs Pulse Rate",
    labels={"steps": "Steps", "pulse_rate": "Pulse Rate"}
)

st.plotly_chart(fig)
