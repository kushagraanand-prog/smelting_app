import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load pre-trained models
# -----------------------------
# ANN model (Keras)
model = tf.keras.models.load_model("ann_model.h5")

# Scaler (used for ANN input normalization)
scaler = joblib.load("scaler.pkl")

# Polynomial regression model
poly = joblib.load("poly_model.pkl")
linreg = joblib.load("linreg_model.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("⚒️ Smelting Process Decision Support Tool")
st.write("Interactive prediction of **Cu in slag** based on process parameters.")

# Sliders for key parameters

Cu = float(st.text_input("Cu", '25') )
Fe = float(st.text_input("Fe",'27'))
S = float(st.text_input("S", '28'))
SiO2 = float(st.text_input("SiO2", '0.6'))
Al2O3 = float(st.text_input("Al2O3", '0.02'))
CaO = float(st.text_input("CaO", '1.3'))
MgO = float(st.text_input("MgO", '1.7'))
S_Cu_ratio = float(st.text_input("S / Cu", '1.13'))
CONC_FEED_RATE = float(st.text_input("CONC. FEED RATE", '100'))
COAL_FEED_RATE = float(st.text_input("COAL FEED RATE", '2'))
C_SLAG_FEED_RATE = float(st.text_input("C-SLAG FEED RATE - S FURNACE", '9'))
S_FURNACE_AIR = float(st.text_input("S-FURNACE AIR", '18000'))
S_FURNACE_OXYGEN = float(st.text_input("S-FURNACE OXYGEN", '170000'))
S_MELT_TEMPERATURE = float(st.text_input("S MELT TEMPERATURE", '1200'))
CL_SLAG_TEMPERATURE = float(st.text_input("Fe/SiO₂ Ratio", '1230'))
Fe_in_CL_slag = float(st.text_input("Fe%_in_CL_slag", '44'))
SiO2_in_CL_slag = float(st.text_input("SiO2%_in_CL_slag", '35'))
CaO_in_CL_slag = float(st.text_input("CaO%_in_CL_slag", '2.2'))
Al2O3_in_CL_slag = float(st.text_input("Al2O3%_in_CL_slag", '4.5'))
MgO_in_CL_slag = float(st.text_input("MgO%_in_CL_slag", '1.4'))
Fe_SiO2_RATIO = float(st.text_input("Fe/SiO2 RATIO", '1.15'))
CLS_Fe3O4 = float(st.text_input("CLS %Fe3O4", '1.7'))
S_outlet_Fe3O4 = float(st.text_input("S-O/L %Fe3O4", '7'))
Cu_in_CL_matte = float(st.text_input("Cu%_in_CL_matte", '69'))
Cu_in_C_slag  = float(st.text_input("Cu%_in_C_slag", '14'))
Fe_in_C_slag  = float(st.text_input("Fe%_in_C_slag", '45'))
CaO_in_C_slag  = float(st.text_input("CaO%_in_C_slag", '16'))
Fe3O4_in_C_slag  = float(st.text_input("%Fe3O4_in_C_slag", '28'))
SILICA_FEED_RATE = float(st.text_input("SILICA FEED RATE", '13'))
# -----------------------------
# ANN Prediction
# -----------------------------
X_input = np.array([[Cu,
Fe,
S,
SiO2,
Al2O3,
CaO,
MgO,
S_Cu_ratio,
CONC_FEED_RATE,
COAL_FEED_RATE,
C_SLAG_FEED_RATE,
S_FURNACE_AIR,
S_FURNACE_OXYGEN,
S_MELT_TEMPERATURE,
CL_SLAG_TEMPERATURE,
Fe_in_CL_slag,
SiO2_in_CL_slag,
CaO_in_CL_slag,
Al2O3_in_CL_slag,
MgO_in_CL_slag,
Fe_SiO2_RATIO,
CLS_Fe3O4,
S_outlet_Fe3O4,
Cu_in_CL_matte,
Cu_in_C_slag,
Fe_in_C_slag,
CaO_in_C_slag,
Fe3O4_in_C_slag,
SILICA_FEED_RATE]])
X_scaled = scaler.transform(X_input)
cu_slag_ann = model.predict(X_scaled).flatten()[0]

# Polynomial Prediction
X_poly = poly.transform(np.array([[Fe_SiO2_RATIO, Cu_in_CL_matte]]))
cu_slag_poly = linreg.predict(X_poly)[0]

# -----------------------------
# Display Results
# -----------------------------
st.subheader("Predicted Cu in Slag (%)")
st.write(f"🔹 ANN Prediction: **{cu_slag_ann:.2f}%**")
st.write(f"🔹 Polynomial Approximation: **{cu_slag_poly:.2f}%**")

# -----------------------------
# 3D Surface Visualization
# -----------------------------
x_range = np.linspace(0.5, 3.0, 30)
y_range = np.linspace(40, 70, 30)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

XY_poly = poly.transform(np.column_stack([X_grid.ravel(), Y_grid.ravel()]))
Z_grid = linreg.predict(XY_poly).reshape(X_grid.shape)

fig = go.Figure(data=[go.Surface(x=X_grid, y=Y_grid, z=Z_grid, colorscale="Viridis")])
fig.update_layout(
    scene=dict(
        xaxis_title="Fe/SiO₂ Ratio",
        yaxis_title="Cu% in Matte",
        zaxis_title="Cu in CL Slag (%)"
    ),
    title="Polynomial Fit Surface: Cu in Slag"
)

st.plotly_chart(fig)
