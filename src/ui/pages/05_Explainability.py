"""Explainability page."""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")

st.title("🔍 Model Explainability")

st.markdown("""
Understand model predictions using **SHAP** (SHapley Additive exPlanations)
""")

model_type = st.selectbox("Select model", ["QSAR", "Toxicity", "GAN"])

if st.button("📊 Generate SHAP Analysis", type="primary"):
    with st.spinner("Computing SHAP values..."):
        import time
        time.sleep(1.5)
        
        # Mock SHAP values
        features = [f"Feature_{i}" for i in range(1, 21)]
        shap_values = np.random.randn(20)
        shap_values = np.abs(shap_values)
        
        # Sort by importance
        indices = np.argsort(shap_values)[::-1]
        top_features = [features[i] for i in indices[:10]]
        top_values = shap_values[indices[:10]]
        
        st.success("✓ Analysis complete!")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top_features)), top_values, color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'{model_type} Model - Feature Importance')
        ax.invert_yaxis()
        st.pyplot(fig)
        
        # Explanation
        st.subheader("Key Insights")
        st.markdown(f"""
        - **Most important feature**: {top_features[0]} (SHAP value: {top_values[0]:.3f})
        - **Top 3 features** account for {(top_values[:3].sum() / top_values.sum() * 100):.1f}% of importance
        - **Model relies** primarily on molecular descriptors and fingerprint bits
        """)
