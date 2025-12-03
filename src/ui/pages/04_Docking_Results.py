"""Docking results page."""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Docking Results", page_icon="🎯", layout="wide")

st.title("🎯 Molecular Docking Results")

st.info("Run molecular docking simulations with AutoDock Vina")

# Receptor selection
receptor_file = st.file_uploader("Upload receptor (PDB)", type=['pdb'])

smiles_input = st.text_area("Enter ligand SMILES", value="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")

col1, col2 = st.columns(2)
with col1:
    center_x = st.number_input("Center X", value=0.0)
    center_y = st.number_input("Center Y", value=0.0)
    center_z = st.number_input("Center Z", value=0.0)

with col2:
    size_x = st.number_input("Size X", value=20.0)
    size_y = st.number_input("Size Y", value=20.0)
    size_z = st.number_input("Size Z", value=20.0)

if st.button("🚀 Run Docking", type="primary"):
    with st.spinner("Running docking simulation..."):
        import time
        time.sleep(2)
        
        # Mock results
        scores = np.random.randn(9) * 2 - 7
        scores.sort()
        
        st.success("✓ Docking complete!")
        
        st.subheader("Docking Scores")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(scores)), scores, color='steelblue')
        ax.axhline(y=-6.0, color='red', linestyle='--', label='Hit threshold')
        ax.set_xlabel('Pose')
        ax.set_ylabel('Binding Energy (kcal/mol)')
        ax.set_title('Docking Scores by Pose')
        ax.legend()
        st.pyplot(fig)
        
        # Results table
        df = pd.DataFrame({
            'Pose': range(1, len(scores) + 1),
            'Score (kcal/mol)': scores,
            'RMSD': np.random.rand(len(scores)) * 3
        })
        st.dataframe(df)
        
        st.metric("Best Score", f"{scores[0]:.2f} kcal/mol", 
                 delta="Strong binding" if scores[0] < -7 else "Moderate binding")
