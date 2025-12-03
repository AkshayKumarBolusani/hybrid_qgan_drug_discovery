"""QSAR and toxicity prediction page."""
import streamlit as st
import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

st.set_page_config(page_title="QSAR & Toxicity", page_icon="🔬", layout="wide")

st.title("🔬 QSAR & Toxicity Prediction")

# Sidebar info
st.sidebar.header("Model Information")
qsar_path = Path("experiments/checkpoints/qsar_rf.pkl")
tox_path = Path("experiments/checkpoints/toxicity_rf.pkl")

qsar_status = "✅ Loaded" if qsar_path.exists() else "❌ Not trained"
tox_status = "✅ Loaded" if tox_path.exists() else "❌ Not trained"

st.sidebar.markdown(f"""
**QSAR Model**: {qsar_status}  
**Toxicity Model**: {tox_status}

ℹ️ Models must be trained first using `train_all_pipeline.py` or the Training page.
""")

@st.cache_resource
def load_qsar_model():
    """Load trained QSAR model from checkpoint."""
    try:
        from src.models.qsar.qsar_sklearn import QSARSklearnModel
        model = QSARSklearnModel(model_type='rf')
        model.load(str(qsar_path))
        return model
    except Exception as e:
        st.error(f"Failed to load QSAR model: {e}")
        return None

@st.cache_resource
def load_toxicity_model():
    """Load trained Toxicity model from checkpoint."""
    try:
        with open(tox_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Failed to load Toxicity model: {e}")
        return None

def compute_fingerprints(smiles_list):
    """Generate molecular fingerprints for model input."""
    try:
        from src.features.fingerprint_utils import batch_smiles_to_fingerprints
        fps = batch_smiles_to_fingerprints(smiles_list, fp_type='morgan', n_bits=2048)
        return np.array(fps)
    except Exception as e:
        st.error(f"Fingerprint generation failed: {e}")
        return None

def compute_rdkit_descriptors(smiles):
    """Compute RDKit molecular descriptors."""
    try:
        from rdkit import Chem
        from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            "qed": QED.qed(mol),
            "logp": Crippen.MolLogP(mol),
            "sa": Descriptors.NumHDonors(mol),  # Simple SA proxy
            "mol_weight": Descriptors.MolWt(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
        }
    except Exception as e:
        st.warning(f"Descriptor computation failed: {e}")
        return None

# Input (preserve user text via session_state)
DEFAULT_SMILES = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C"
if "smiles_input" not in st.session_state:
    # Initialize once before the widget is created
    st.session_state["smiles_input"] = DEFAULT_SMILES

st.text_area(
    "Enter SMILES (one per line)",
    value=st.session_state["smiles_input"],
    height=150,
    key="smiles_input"
)

col1, col2 = st.columns([1, 4])
with col1:
    predict_button = st.button("🎯 Predict Properties", type="primary")

if predict_button:
    # Read current input from session_state; do not assign to it after widget instantiation
    smiles_text = st.session_state.get("smiles_input", DEFAULT_SMILES)
    smiles_list = [s.strip() for s in smiles_text.split('\n') if s.strip()]
    
    if not smiles_list:
        st.warning("Please enter at least one SMILES string.")
        st.stop()
    
    st.success(f"✓ Analyzing {len(smiles_list)} molecules...")
    
    # Load models if available
    qsar_model = load_qsar_model() if qsar_path.exists() else None
    tox_model = load_toxicity_model() if tox_path.exists() else None
    
    # Generate fingerprints once for all molecules
    fps = None
    if qsar_model or tox_model:
        fps = compute_fingerprints(smiles_list)
    
    # Predict properties
    qsar_preds = None
    tox_preds = None
    
    if qsar_model and fps is not None:
        try:
            qsar_preds = qsar_model.predict(fps)
        except Exception as e:
            st.error(f"QSAR prediction failed: {e}")
    
    if tox_model and fps is not None:
        try:
            tox_preds = tox_model.predict(fps)
        except Exception as e:
            st.error(f"Toxicity prediction failed: {e}")
    
    # Display results per molecule
    for idx, smiles in enumerate(smiles_list, 1):
        st.markdown(f"### Molecule {idx}")
        st.code(smiles, language=None)
        
        descriptors = compute_rdkit_descriptors(smiles)
        
        tab1, tab2, tab3 = st.tabs(["📊 Descriptors", "☠️ Toxicity", "💊 ADMET"])
        
        with tab1:
            if descriptors:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("QED", f"{descriptors['qed']:.3f}")
                with col2:
                    st.metric("LogP", f"{descriptors['logp']:.2f}")
                with col3:
                    st.metric("TPSA", f"{descriptors['tpsa']:.1f}")
                with col4:
                    st.metric("MW", f"{int(descriptors['mol_weight'])}")
                
                # Show QSAR prediction if available
                if qsar_preds is not None:
                    st.markdown("#### QSAR Prediction")
                    st.metric("Predicted Solubility (LogS)", f"{qsar_preds[idx-1]:.3f}")
            else:
                st.warning("⚠️ Invalid SMILES - cannot compute descriptors")
        
        with tab2:
            if tox_preds is not None and idx <= len(tox_preds):
                st.warning("⚠️ Toxicity Assessment")
                
                # Tox21 task names (12 tasks)
                tox_tasks = [
                    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
                    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
                    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
                ]
                
                mol_tox = tox_preds[idx-1]
                
                for task_idx, task_name in enumerate(tox_tasks[:len(mol_tox)]):
                    prob = mol_tox[task_idx]
                    
                    if prob < 0.3:
                        risk = "Low"
                        color = "🟢"
                    elif prob < 0.7:
                        risk = "Medium"
                        color = "🟡"
                    else:
                        risk = "High"
                        color = "🔴"
                    
                    st.markdown(f"{color} **{task_name}**: {risk} Risk (Score: {prob:.3f})")
            else:
                st.info("ℹ️ Toxicity model not available. Train the model first using `train_all_pipeline.py`.")
        
        with tab3:
            if descriptors:
                st.markdown(f"""
                - **Molecular Weight**: {descriptors['mol_weight']:.1f} Da
                - **LogP**: {descriptors['logp']:.2f}
                - **TPSA**: {descriptors['tpsa']:.1f} Ų
                - **H-Bond Donors**: {descriptors['hbd']}
                - **H-Bond Acceptors**: {descriptors['hba']}
                - **QED (Drug-likeness)**: {descriptors['qed']:.3f}
                
                **Lipinski's Rule of Five**:
                - MW < 500: {"✅" if descriptors['mol_weight'] < 500 else "❌"}
                - LogP < 5: {"✅" if descriptors['logp'] < 5 else "❌"}
                - HBD ≤ 5: {"✅" if descriptors['hbd'] <= 5 else "❌"}
                - HBA ≤ 10: {"✅" if descriptors['hba'] <= 10 else "❌"}
                """)
            else:
                st.warning("⚠️ Cannot compute ADMET properties for invalid SMILES")
        
        st.markdown("---")
