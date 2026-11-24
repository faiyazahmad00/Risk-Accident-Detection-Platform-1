"""
Risk Accident Detection Platform
Beginner-friendly demonstration using synthetic training data.

IMPORTANT:
This is an educational demo, not a validated road-safety prediction system.
The generated data and resulting scores are synthetic and must not be used
for real-world driving, emergency, insurance, or traffic-control decisions.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Risk Accident Detection Platform",
    page_icon="🚦",
    layout="wide"
)

FEATURES = [
    "weather", "road_surface", "lighting", "traffic_density",
    "hour", "speed_limit", "road_curvature"
]

def create_demo_data(rows=2500, random_state=42):
    """Create illustrative synthetic data for demonstrating the workflow."""
    rng = np.random.default_rng(random_state)

    weather = rng.choice(["Clear", "Rain", "Fog", "Snow"], rows,
                         p=[0.55, 0.20, 0.15, 0.10])
    road_surface = rng.choice(["Good", "Average", "Poor"], rows,
                              p=[0.50, 0.35, 0.15])
    lighting = rng.choice(["Daylight", "Night", "Twilight"], rows,
                          p=[0.55, 0.30, 0.15])
    traffic_density = rng.choice(["Low", "Medium", "High"], rows,
                                 p=[0.30, 0.45, 0.25])
    hour = rng.integers(0, 24, rows)
    speed_limit = rng.choice([30, 40, 50, 60, 80, 100], rows)
    road_curvature = rng.choice(["Straight", "Moderate", "Sharp"], rows,
                                p=[0.55, 0.30, 0.15])

    # Illustrative scoring rules only; these are not real-world risk estimates.
    score = np.full(rows, 10.0)
    score += np.where(weather == "Rain", 12, 0)
    score += np.where(weather == "Fog", 18, 0)
    score += np.where(weather == "Snow", 22, 0)
    score += np.where(road_surface == "Average", 7, 0)
    score += np.where(road_surface == "Poor", 16, 0)
    score += np.where(lighting == "Night", 13, 0)
    score += np.where(lighting == "Twilight", 6, 0)
    score += np.where(traffic_density == "Medium", 7, 0)
    score += np.where(traffic_density == "High", 15, 0)
    score += np.where((hour >= 0) & (hour <= 5), 8, 0)
    score += np.where(speed_limit >= 80, 8, 0)
    score += np.where(road_curvature == "Moderate", 6, 0)
    score += np.where(road_curvature == "Sharp", 13, 0)
    score += rng.normal(0, 4, rows)
    score = np.clip(score, 0, 100)

    data = pd.DataFrame({
        "weather": weather,
        "road_surface": road_surface,
        "lighting": lighting,
        "traffic_density": traffic_density,
        "hour": hour,
        "speed_limit": speed_limit,
        "road_curvature": road_curvature,
        "risk_score": score
    })
    return data


@st.cache_resource
def train_demo_model():
    """Train a small model once and cache it for the Streamlit session."""
    data = create_demo_data()
    X = pd.get_dummies(data[FEATURES], columns=[
        "weather", "road_surface", "lighting",
        "traffic_density", "road_curvature"
    ])
    y = data["risk_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,
        random_state=42,
        max_depth=12,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions)
    }
    return model, list(X.columns), metrics


def risk_label(score):
    if score < 35:
        return "Low", "🟢"
    elif score < 65:
        return "Moderate", "🟠"
    return "High", "🔴"


def main():
    st.title("🚦 Risk Accident Detection Platform")
    st.caption("Educational machine-learning demonstration using synthetic data.")

    st.warning(
        "Demo limitation: this app trains on artificially generated data. "
        "Its score is not a real-world accident probability and must not be "
        "used to make driving or safety decisions."
    )

    model, model_columns, metrics = train_demo_model()

    st.sidebar.header("Road and Environment Details")
    weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Fog", "Snow"])
    road_surface = st.sidebar.selectbox("Road surface", ["Good", "Average", "Poor"])
    lighting = st.sidebar.selectbox("Lighting", ["Daylight", "Night", "Twilight"])
    traffic_density = st.sidebar.selectbox(
        "Traffic density", ["Low", "Medium", "High"]
    )
    hour = st.sidebar.slider("Hour of day (0–23)", 0, 23, 14)
    speed_limit = st.sidebar.selectbox(
        "Road speed-limit context (km/h)", [30, 40, 50, 60, 80, 100], index=3
    )
    road_curvature = st.sidebar.selectbox(
        "Road curvature", ["Straight", "Moderate", "Sharp"]
    )

    input_data = pd.DataFrame([{
        "weather": weather,
        "road_surface": road_surface,
        "lighting": lighting,
        "traffic_density": traffic_density,
        "hour": hour,
        "speed_limit": speed_limit,
        "road_curvature": road_curvature
    }])

    encoded_input = pd.get_dummies(input_data, columns=[
        "weather", "road_surface", "lighting",
        "traffic_density", "road_curvature"
    ])
    encoded_input = encoded_input.reindex(columns=model_columns, fill_value=0)

    score = float(np.clip(model.predict(encoded_input)[0], 0, 100))
    label, icon = risk_label(score)

    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Demo risk score", f"{score:.1f} / 100")
    col2.metric("Demo risk category", f"{icon} {label}")
    col3.metric("Model", "Random Forest")

    st.progress(int(round(score)))
    st.info(
        "Interpretation: this is a score produced by the demonstration model "
        "from synthetic training examples. It is not a calibrated probability."
    )

    with st.expander("View selected input values"):
        st.dataframe(input_data, use_container_width=True)

    with st.expander("Demo model evaluation"):
        st.write("Evaluation on a held-out portion of the synthetic dataset:")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{metrics['mae']:.2f}")
        m2.metric("RMSE", f"{metrics['rmse']:.2f}")
        m3.metric("R²", f"{metrics['r2']:.3f}")
        st.caption(
            "These metrics measure performance on synthetic demo data only. "
            "They do not establish real-world predictive accuracy."
        )

    st.subheader("How the application works")
    st.markdown(
        """
        1. Choose weather, road, lighting and traffic conditions in the sidebar.
        2. The application converts the selections into model input features.
        3. A Random Forest model predicts a demonstration score.
        4. The score is displayed as Low, Moderate or High.
        """
    )
    st.caption(
        "For a real research project, replace the synthetic dataset with a "
        "reliable, documented road-safety dataset; validate the model carefully "
        "before making any claims about predictive performance."
    )


if __name__ == "__main__":
    main()
