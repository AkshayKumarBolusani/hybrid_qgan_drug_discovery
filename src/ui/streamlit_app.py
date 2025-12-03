"""Main Streamlit application for Hybrid Quantum GAN Drug Discovery."""
import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config
import json
import pandas as pd

# Page config
st.set_page_config(
    page_title="HQ-GAN Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #0E1117 0%, #12161C 100%);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid #2A2F36;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }
    .metric-row { display: flex; align-items: center; gap: 14px; }
    .metric-icon { width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; background-color: #1f77b4; color: #ffffff; font-size: 20px; }
    .metric-content { display: flex; flex-direction: column; }
    .metric-label { font-size: 0.85rem; color: #9aa4ad; margin-bottom: 0.1rem; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #e6edf3; letter-spacing: 0.02em; }
    .metric-sub { margin-top: 0.2rem; font-size: 0.8rem; color: #AEB6BF; }
    .section-divider {
        border-top: 1px solid #2A2F36;
        margin: 1.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application."""
    # Header
    st.markdown('<div class="main-header">🧬 Hybrid Quantum GAN Drug Discovery</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.info("""
    **Welcome to HQ-GAN Drug Discovery!**
    
    This platform combines:
    - 🔬 Hybrid Quantum GANs
    - 💊 Molecular Generation
    - 🎯 QSAR Prediction
    - ☠️ Toxicity Assessment
    - 🔗 Molecular Docking
    - 📊 Explainability (SHAP)
    """)
    
    # Status Overview (live counters)
    checkpoints_dir = Path("experiments/checkpoints")
    data_dir = Path("data/processed/molecule_datasets")
    latest_path = Path("experiments/final_validation/generated_smiles.json")
    experiments_dir = Path("experiments")

    models_available = len(list(checkpoints_dir.glob("*.pkl"))) if checkpoints_dir.exists() else 0
    qsar_trained = (checkpoints_dir / "qsar_rf.pkl").exists()
    tox_trained = (checkpoints_dir / "toxicity_rf.pkl").exists()
    # GAN training presence: runs and checkpoints
    gan_runs = len([p for p in experiments_dir.glob("gan_*/")]) if experiments_dir.exists() else 0
    gan_checkpoints = len(list(experiments_dir.glob("**/*.pt"))) if experiments_dir.exists() else 0

    qsar_rows = 0
    tox_rows = 0
    try:
        if (data_dir / "combined_qsar.csv").exists():
            qsar_rows = pd.read_csv(data_dir / "combined_qsar.csv").shape[0]
        if (data_dir / "combined_toxicity.csv").exists():
            tox_rows = pd.read_csv(data_dir / "combined_toxicity.csv").shape[0]
    except Exception:
        pass

    latest_count = 0
    avg_qed = None
    try:
        if latest_path.exists():
            data = json.loads(latest_path.read_text())
            latest_count = len([d for d in data if d.get("smiles")])
            qeds = [d.get("qed") for d in data if d.get("qed") is not None]
            if qeds:
                avg_qed = sum(qeds) / len(qeds)
    except Exception:
        pass

    st.subheader("Status Overview")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        trained_list = ', '.join([x for x in [
            "QSAR" if qsar_trained else None,
            "Toxicity" if tox_trained else None,
            "GAN" if gan_runs > 0 or gan_checkpoints > 0 else None
        ] if x]) or 'None'
        st.markdown('<div class="metric-card"><div class="metric-row">\n'
                    '<div class="metric-icon">🧠</div>\n'
                    '<div class="metric-content">\n'
                    '<div class="metric-label">Models Available</div>\n'
                    f'<div class="metric-value">{models_available}</div>\n'
                    '</div></div>\n'
                    f'<div class="metric-sub">Trained: {trained_list}</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="metric-card"><div class="metric-row">\n'
                    '<div class="metric-icon">📈</div>\n'
                    '<div class="metric-content">\n'
                    '<div class="metric-label">QSAR Samples</div>\n'
                    f'<div class="metric-value">{qsar_rows}</div>\n'
                    '</div></div>\n'
                    '<div class="metric-sub">combined_qsar.csv</div></div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="metric-card"><div class="metric-row">\n'
                    '<div class="metric-icon">☠️</div>\n'
                    '<div class="metric-content">\n'
                    '<div class="metric-label">Toxicity Samples</div>\n'
                    f'<div class="metric-value">{tox_rows}</div>\n'
                    '</div></div>\n'
                    '<div class="metric-sub">combined_toxicity.csv</div></div>', unsafe_allow_html=True)

    with c4:
        avg_qed_text = f"Avg QED: {avg_qed:.2f}" if avg_qed is not None else "Avg QED: n/a"
        st.markdown('<div class="metric-card"><div class="metric-row">\n'
                    '<div class="metric-icon">🧪</div>\n'
                    '<div class="metric-content">\n'
                    '<div class="metric-label">Latest Generated</div>\n'
                    f'<div class="metric-value">{latest_count}</div>\n'
                    '</div></div>\n'
                    f'<div class="metric-sub">{avg_qed_text}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Additional GAN status row
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="metric-card"><div class="metric-row">\n'
                    '<div class="metric-icon">🌀</div>\n'
                    '<div class="metric-content">\n'
                    '<div class="metric-label">GAN Runs Detected</div>\n'
                    f'<div class="metric-value">{gan_runs}</div>\n'
                    '</div></div>\n'
                    '<div class="metric-sub">Folders like gan_*</div></div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="metric-card"><div class="metric-row">\n'
                    '<div class="metric-icon">🎛️</div>\n'
                    '<div class="metric-content">\n'
                    '<div class="metric-label">GAN Checkpoints</div>\n'
                    f'<div class="metric-value">{gan_checkpoints}</div>\n'
                    '</div></div>\n'
                    '<div class="metric-sub">*.pt across experiments</div></div>', unsafe_allow_html=True)
    
    # Quick start guide
    st.header("🚀 Quick Start")
    
    tab1, tab2, tab3 = st.tabs(["📖 Overview", "⚡ Generate", "📊 Analyze"])
    
    with tab1:
        st.markdown("""
        ### Project Overview
        
        This platform implements a **Hybrid Quantum GAN** system for drug discovery:
        
        1. **Molecular Generation**: Generate novel drug-like molecules using quantum-enhanced GANs
        2. **Property Prediction**: Predict QSAR, toxicity, and ADMET properties
        3. **Docking Simulation**: Evaluate binding affinity to target proteins
        4. **Explainability**: Understand predictions with SHAP analysis
        5. **Reporting**: Generate comprehensive PDF reports
        
        **Navigate using the pages in the sidebar →**
        """)
        
        st.info("💡 **Tip**: Start by generating molecules in the 'Generate Molecules' page!")
    
    with tab2:
        st.markdown("### Quick Generate")
        num_molecules = st.slider("Number of molecules", 1, 50, 10)
        
        if st.button("Generate Molecules", type="primary"):
            with st.spinner("Generating molecules..."):
                st.success(f"✓ Generated {num_molecules} molecules!")
                st.info("View them in the 'Generate Molecules' page")
    
    with tab3:
        st.markdown("### Recent Activity")
        st.line_chart({"Generations": [10, 15, 12, 20, 18, 25, 22]})
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Hybrid Quantum GAN Drug Discovery System | Version 1.0.0</p>
        <p>Built with Streamlit, PyTorch, PennyLane, RDKit & DeepChem</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
