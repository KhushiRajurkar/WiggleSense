import streamlit as st
import pandas as pd
import numpy as np
from math import atan2, degrees
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="WiggleSense: Bee Dance Decoder", layout="centered", page_icon="🐝")
st.title("\U0001F41D WiggleSense: Real-Time Bee Dance Decoder")
st.sidebar.markdown("### 📥 Upload bee pose data to decode wiggle behavior!")

# --- Helper functions ---
def compute_angle(dx, dy):
    return degrees(atan2(dy, dx)) % 360

def detect_wiggle(angle_series, var_window=5, threshold=30):
    smooth = angle_series.rolling(window=3, center=True).mean()
    variance = smooth.rolling(window=var_window, center=True).var()
    return variance < threshold, variance

def health_score(angle_var):
    if pd.isna(angle_var):
        return "🟡 Low activity"
    elif angle_var < 30:
        return "✅ Healthy"
    elif angle_var > 100:
        return "⚠️ Stress"
    else:
        return "🟡 Low activity"

# --- Upload Section ---
st.sidebar.header("Upload CSV with Pose Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check required columns
    required_cols = {"dx", "dy", "image", "bee_id", "frame_id"}
    if not required_cols.issubset(set(df.columns)):
        st.error(f"CSV must contain the following columns: {', '.join(required_cols)}")
    else:
        st.success("CSV successfully loaded!")

        # Compute angle
        df["angle_deg"] = df.apply(lambda row: compute_angle(row["dx"], row["dy"]), axis=1)

        # Wiggle detection
        df = df.sort_values(["bee_id", "frame_id"])
        df["angle_smooth"] = df.groupby("bee_id")["angle_deg"].transform(lambda x: x.rolling(3, center=True).mean())
        df["angle_var"] = df.groupby("bee_id")["angle_smooth"].transform(lambda x: x.rolling(5, center=True).var())
        df["wiggle"], _ = detect_wiggle(df["angle_deg"])
        df["hive_health_score"] = df["angle_var"].apply(health_score)

        # Convert wiggle to emoji for display
        df["wiggle_display"] = df["wiggle"].apply(lambda x: "✅" if x else "❌")

        # --- Output Section ---
        st.subheader("Sample Predictions")
        st.dataframe(df[["bee_id", "frame_id", "angle_deg", "wiggle_display", "hive_health_score"]].head(10))

        # --- Plots ---
        st.subheader("Body Angle Over Time")
        sample_bee = df["bee_id"].value_counts().idxmax()
        df_sample = df[df["bee_id"] == sample_bee]

        fig, ax = plt.subplots()
        ax.plot(df_sample["frame_id"], df_sample["angle_deg"], label="Angle")
        ax.plot(df_sample["frame_id"], df_sample["angle_smooth"], label="Smoothed", linestyle="--")
        ax.set_title(f"Bee ID {sample_bee} - Body Angle")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Wiggle Detection")
        fig2, ax2 = plt.subplots()
        ax2.plot(df_sample["frame_id"], df_sample["angle_var"], label="Variance", color="orange")
        ax2.axhline(30, color='red', linestyle='--', label='Wiggle Threshold')
        ax2.set_title("Wiggle Loop Detection")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Variance")
        ax2.legend()
        st.pyplot(fig2)

        # --- Downloadable Output ---
        st.subheader("Download Processed Data")
        csv_out = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv_out, "wigglesense_output.csv", "text/csv")
