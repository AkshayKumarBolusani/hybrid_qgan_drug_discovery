"""Molecule generation page."""
import streamlit as st
import sys
import json
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path
from io import BytesIO
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

st.set_page_config(page_title="Generate Molecules", page_icon="🎲", layout="wide")

st.title("🎲 Generate Molecules")

# Sidebar controls
st.sidebar.header("Generation Settings")

num_molecules = st.sidebar.slider("Number of molecules", 1, 100, 10)
temperature = st.sidebar.slider("Sampling temperature", 0.1, 2.0, 1.0, 0.1)
use_quantum = st.sidebar.checkbox("Use quantum layers", value=True)
filter_druglike = st.sidebar.checkbox("Filter drug-like only", value=True)
use_latest = st.sidebar.checkbox("Use latest generated outputs", value=False,
                                 help="Load `experiments/final_validation/generated_smiles.json` if available")

st.sidebar.markdown("---")
st.sidebar.info("""
**Temperature**: Controls diversity
- Low (< 1.0): More conservative
- High (> 1.0): More diverse

**Quantum Layers**: Enable/disable quantum enhancement
""")

def draw_molecule_grid(smiles_list, labels=None):
    try:
        from src.utils.visualization_utils import plot_molecule_grid
        img = plot_molecule_grid(smiles_list, labels=labels, mols_per_row=4, img_size=(250, 250))
        if img is not None:
            st.image(img, caption="Molecule grid", use_column_width=True)
            return True
    except Exception as e:
        st.warning(f"Molecule rendering failed: {e}")
    return False

def draw_single(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.warning("Invalid SMILES; showing placeholder")
            return False
        img = Draw.MolToImage(mol, size=(300, 300))
        st.image(img, caption=smiles)
        return True
    except Exception as e:
        st.warning(f"RDKit draw error: {e}")
        return False

def build_pdf(smiles_with_metrics):
    """Generate a styled PDF containing molecule images and metrics.
    Returns BytesIO of the PDF.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.units import inch
        from rdkit import Chem
        from rdkit.Chem import Draw

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, title="Hybrid QGAN Drug Discovery Report")
        styles = getSampleStyleSheet()

        story = []

        # Header
        title = Paragraph("<b>Hybrid QGAN Drug Discovery</b>", styles["Title"])
        subtitle = Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"])
        story.extend([title, subtitle, Spacer(1, 0.2*inch)])

        # Per-molecule sections
        for i, row in enumerate(smiles_with_metrics, 1):
            smi = row.get("smiles", "")
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            img = Draw.MolToImage(mol, size=(300, 300))
            img_buf = BytesIO()
            img.save(img_buf, format='PNG')
            img_buf.seek(0)

            story.append(Paragraph(f"<b>Molecule {i}</b>", styles["Heading2"]))
            story.append(Paragraph(f"SMILES: <font face='Courier'>{smi}</font>", styles["Normal"]))
            story.append(Image(img_buf, width=2.5*inch, height=2.5*inch))

            # Metrics table
            tbl_data = [["Metric", "Value"]]
            for key in ("qed", "logp", "mol_weight", "tpsa"):
                val = row.get(key)
                if val is None:
                    continue
                if isinstance(val, float):
                    display = f"{val:.3f}" if key in ("qed", "logp") else f"{val:.1f}"
                else:
                    display = str(val)
                tbl_data.append([key.upper(), display])

            table = Table(tbl_data, colWidths=[1.5*inch, 2.0*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1B1F24')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#E0E0E0')),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2A2F36')),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#0E1117')),
                ('TEXTCOLOR', (0,1), (-1,-1), colors.white),
                ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ]))
            story.extend([table, Spacer(1, 0.2*inch)])

        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")
        return None

def copy_to_clipboard(smiles: str, key: str):
    """Render a small 'Copy' button that copies SMILES and shows a status message."""
    try:
        import json as _json
        safe = _json.dumps(smiles)
        components.html(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-top:6px;">
          <button id="btn-{key}" style="padding:6px 10px;border:1px solid #2A2F36;border-radius:6px;background:#1B1F24;color:#E0E0E0;cursor:pointer;">Copy</button>
          <span id="msg-{key}" style="font-size:0.9rem;color:#9aa4ad;"></span>
        </div>
        <script>
        const btn = document.getElementById("btn-{key}");
        const msg = document.getElementById("msg-{key}");
        btn.onclick = async () => {{
          try {{
            await navigator.clipboard.writeText({safe});
            msg.textContent = "Copied " + {safe} + " to clipboard.";
            msg.style.color = "#22c55e";
          }} catch (e) {{
            msg.textContent = "Copy failed.";
            msg.style.color = "#ef4444";
          }}
        }};
        </script>
        """, height=40)
    except Exception:
        st.caption("Copy not available in this environment")

# Main content
if st.button("🚀 Generate Molecules", type="primary") and not use_latest:
    progress_bar = st.progress(0)
    status = st.empty()
    
    # Simulate generation (mock until training generation wired)
    import time
    import random
    from rdkit import Chem
    from rdkit.Chem import QED, Crippen, rdMolDescriptors, Descriptors

    def compute_metrics(smi: str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        def safe(fn, default=None):
            try:
                return float(fn(mol))
            except Exception:
                return default
        return {
            "smiles": smi,
            "qed": safe(QED.qed),
            "logp": safe(Crippen.MolLogP),
            "sa": safe(Descriptors.NumHDonors),  # simple SA proxy
            "mol_weight": safe(Descriptors.MolWt),
            "tpsa": safe(rdMolDescriptors.CalcTPSA),
        }

    generated_smiles = []
    for i in range(num_molecules):
        progress_bar.progress((i + 1) / num_molecules)
        status.text(f"Generating molecule {i+1}/{num_molecules}...")
        time.sleep(0.05)
        mock_smiles = random.choice([
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'CC(=O)OC1=CC=CC=C1C(=O)O',
            'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F',
            'c1ccccc1C(O)c2ccccc2',
            'CNC(=O)Oc1cccc2ccccc12',
            'CON(C)C(=O)Nc1ccc(Cl)c(Cl)c1',
        ])
        generated_smiles.append(mock_smiles)
    
    progress_bar.empty()
    status.success(f"✓ Generated {num_molecules} molecules!")
    
    # Compute metrics
    metrics = [m for m in (compute_metrics(s) for s in generated_smiles) if m]

    # Persist outputs for Latest section and external use
    out_dir = Path("experiments/final_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generated_smiles.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(metrics).to_csv(out_dir / "results_table.csv", index=False)

    # Display grid
    st.subheader("Generated Molecules")
    if not draw_molecule_grid([m["smiles"] for m in metrics][:16], labels=[f"QED {m['qed']:.2f}" if m['qed'] is not None else '' for m in metrics][:16]):
        st.image("https://via.placeholder.com/600x300.png?text=Molecule+Grid+Visualization",
                 caption="Molecule visualization placeholder")
    
    # Display list with images and mock metrics
    st.subheader("Details")
    for idx, row in enumerate(metrics[:10], 1):
        with st.expander(f"Molecule {idx}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                _ok = draw_single(row["smiles"])  # draw image
                st.code(row["smiles"])  # always show SMILES alongside image
                copy_to_clipboard(row["smiles"], key=f"gen-{idx}")
            with col2:
                if row.get("qed") is not None:
                    st.metric("QED", f"{row['qed']:.3f}")
                if row.get("logp") is not None:
                    st.metric("LogP", f"{row['logp']:.2f}")
                if row.get("mol_weight") is not None:
                    st.metric("MW", f"{int(row['mol_weight'])}")
    
    # Download button
    smiles_text = '\n'.join([m["smiles"] for m in metrics])
    st.download_button(
        label="📥 Download SMILES",
        data=smiles_text,
        file_name="generated_molecules.txt",
        mime="text/plain"
    )

    # PDF download with images and metrics
    pdf_buf = build_pdf(metrics)
    if pdf_buf:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="🧾 Download PDF Report",
            data=pdf_buf,
            file_name=f"Hybrid_QGAN_Report_{ts}.pdf",
            mime="application/pdf"
        )

    # Metrics table
    st.subheader("Metrics Table")
    st.dataframe(pd.DataFrame(metrics), use_container_width=True)

else:
    # Load latest generated outputs if available
    latest_path = Path("experiments/final_validation/generated_smiles.json")
    st.info("👆 Click the button to generate new molecules or enable 'Use latest generated outputs'.")
    st.subheader("Latest Generated Molecules")
    refresh = st.button("🔄 Refresh latest results", help="Reload saved outputs to update images")
    # Trigger rerun immediately to avoid catching rerun as an exception
    if refresh:
        st.rerun()
    if latest_path.exists():
        try:
            data = json.loads(latest_path.read_text())
            smiles_list = [d.get("smiles") for d in data if d.get("smiles")]
            labels = [f"QED {d.get('qed', 0):.2f}" for d in data]
            # Grid overview
            if not draw_molecule_grid(smiles_list, labels=labels):
                st.image("https://via.placeholder.com/600x300.png?text=Molecule+Grid+Visualization",
                         caption="Molecule visualization placeholder")
            # Per-molecule images and metrics
            st.subheader("Per‑molecule details")
            import pandas as pd
            for i, row in enumerate(data, 1):
                smi = row.get("smiles")
                qed = row.get("qed")
                logp = row.get("logp")
                sa = row.get("sa")
                with st.expander(f"Molecule {i}"):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        _ok = draw_single(smi)  # draw image
                        st.code(smi)  # always show SMILES alongside image
                        copy_to_clipboard(smi, key=f"latest-{i}")
                    with c2:
                        if qed is not None:
                            st.metric("QED", f"{qed:.3f}")
                        if logp is not None:
                            st.metric("LogP", f"{logp:.2f}")
                        if sa is not None:
                            st.metric("SA", f"{sa:.2f}")
            # Metrics table
            st.subheader("Metrics Table")
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            # PDF for latest
            pdf_buf = build_pdf(data)
            if pdf_buf:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="🧾 Download PDF Report",
                    data=pdf_buf,
                    file_name=f"Hybrid_QGAN_Report_{ts}.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.warning(f"Failed to load latest outputs: {e}")
    else:
        st.image("https://via.placeholder.com/600x300.png?text=Molecule+Grid+Visualization",
                 caption="Molecule visualization placeholder")
        st.caption("No generated outputs found yet. Run generation or training.")
