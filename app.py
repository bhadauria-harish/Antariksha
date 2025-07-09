import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and threshold
model = joblib.load('catboost_final.pkl')
with open('optimal_threshold.txt', 'r') as f:
    threshold = float(f.read())

# Feature columns required by the model
feature_columns = ['Speed_bulk_proton', 'Temp', 'Density', 'proton_xvelocity',
       'proton_yvelocity', 'alpha_density', 'alpha_bulk_speed',
       'alpha_thermal', 'Bx_gsm', 'By_gsm', 'Bz_gsm', 'Bx_gse', 'By_gse',
       'Bz_gse', 'mean_integrated_flux_s16_mod',
       'mean_integrated_flux_s19_mod', 'mean_integrated_flux_s9_mod',
       'mean_integrated_flux_s11_mod']

st.title("☀️ Halo CME Event Detector")
st.markdown("### Paste your input data as comma-separated values below:")

example_text = "322.95847206537087,53.03811629704764,5.785019113890819,-322.95847206537087,11.754848539463634,0.3016128240843143,305.2861513691853,72.94986294582651,-3.7569525,2.9090173,-6.580824,-3.7569525,5.852858,-4.1849346,27734126.98412698,88106989.24731185,64486120.67346436,138287344.7299542"

user_input_text = st.text_area("📥 Enter feature values (comma-separated):", value=example_text)

if st.button("Predict CME Event"):
    try:
        # Parse input
        input_values = [float(val.strip()) for val in user_input_text.split(",")]
        
        if len(input_values) != len(feature_columns):
            st.error(f"❌ Expected {len(feature_columns)} values, but got {len(input_values)}.")
        else:
            input_df = pd.DataFrame([input_values], columns=feature_columns)

            # Predict
            proba = model.predict_proba(input_df)[0, 1]
            pred = int(proba > threshold)

            st.write(f"🔍 Predicted Probability of CME: `{proba:.4f}`")

            if pred == 1:
                st.markdown("### ⚠️ **ALERT: Halo CME Event Detected!**")
                st.warning("Hello! A Halo CME event is occurring. Please take the following precautions:")
                st.markdown("""
                - 🛰️ Secure communication and navigation satellites.
                - 🧑‍🚀 Move astronauts to shielded areas.
                - 🔌 Protect sensitive electronics from geomagnetic currents.
                - 📡 Monitor solar activity and space weather forecasts.
                - 🚫 Avoid launching spacecraft during this period.
                """)
            else:
                st.success("✅ No CME event detected based on input data.")
    except ValueError:
        st.error("❌ Invalid input. Please ensure all values are numeric and comma-separated.")
