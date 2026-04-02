"""
app.py
======
Phase 5 — Streamlit Web Application for Multimodal Lung Infection Detection.

HOW TO RUN:
  pip install streamlit torch torchvision pillow numpy pandas joblib scikit-learn matplotlib plotly
  streamlit run app.py

FEATURES:
  - Upload chest X-ray image (JPG/PNG)
  - Enter or skip clinical parameters
  - View prediction with confidence bar
  - See probability breakdown for all 4 classes
  - Grad-CAM heatmap visualisation
  - Download report as JSON
"""

import io
import json
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lung Infection Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #1a6b5a, #2196a8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .subheader { color: #6b7280; font-size: 1rem; margin-bottom: 2rem; }
  .prediction-card {
    border-radius: 12px; padding: 1.5rem;
    border: 2px solid; margin: 1rem 0;
  }
  .pred-NORMAL    { background: #f0fdf4; border-color: #22c55e; color: #14532d; }
  .pred-PNEUMONIA { background: #fef3c7; border-color: #f59e0b; color: #78350f; }
  .pred-TB        { background: #fef2f2; border-color: #ef4444; color: #7f1d1d; }
  .pred-COVID19   { background: #eff6ff; border-color: #3b82f6; color: #1e3a8a; }
  .metric-box {
    background: #f8fafc; border-radius: 8px;
    padding: 1rem; text-align: center; border: 1px solid #e2e8f0;
  }
  .metric-value { font-size: 1.8rem; font-weight: 700; color: #1e293b; }
  .metric-label { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }
  .stProgress > div > div { border-radius: 4px; }
  .info-box {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
    font-size: 0.9rem; color: #1e3a8a; margin: 0.5rem 0;
  }
  .warn-box {
    background: #fffbeb; border-left: 4px solid #f59e0b;
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
    font-size: 0.9rem; color: #78350f; margin: 0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["NORMAL", "PNEUMONIA", "TB", "COVID19"]
CLASS_COLORS = {
    "NORMAL":    "#22c55e",
    "PNEUMONIA": "#f59e0b",
    "TB":        "#ef4444",
    "COVID19":   "#3b82f6",
}
CLASS_ICONS = {
    "NORMAL":    "✅",
    "PNEUMONIA": "⚠️",
    "TB":        "🔴",
    "COVID19":   "🔵",
}
CLASS_DESCRIPTIONS = {
    "NORMAL":    "No significant lung abnormalities detected.",
    "PNEUMONIA": "Bacterial/viral pneumonia pattern detected. Consolidation or infiltrates present.",
    "TB":        "Tuberculosis pattern detected. Possible cavitary lesions or upper lobe involvement.",
    "COVID19":   "COVID-19 pattern detected. Bilateral ground-glass opacities may be present.",
}

CLINICAL_FIELDS = [
    ("age",          "Age",                   18, 90, 40,   1,  "years"),
    ("gender",       "Gender",                0,  1,  1,    1,  "0=F, 1=M"),
    ("temperature",  "Body Temperature",      35.5, 42.0, 37.0, 0.1, "°C"),
    ("oxygen_sat",   "Oxygen Saturation",     70.0, 100.0, 98.0, 0.5, "%"),
    ("wbc_count",    "WBC Count",             2.0, 30.0, 7.0, 0.5, "×10³/μL"),
    ("crp_level",    "CRP Level",             0.1, 300.0, 5.0, 0.5, "mg/L"),
    ("cough_days",   "Cough Duration",        0,   120, 0, 1, "days"),
]
CLINICAL_FLAGS = [
    ("fever",        "Fever"),
    ("dyspnea",      "Shortness of Breath"),
    ("hemoptysis",   "Coughing Blood"),
    ("night_sweats", "Night Sweats"),
    ("weight_loss",  "Unexplained Weight Loss"),
    ("smoker",       "Smoker"),
]


# ─── Model loader (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models …")
def load_predictor(models_dir="./models"):
    try:
        from fusion_model import FusionPredictor
        return FusionPredictor(models_dir=models_dir), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


# ─── Grad-CAM (lightweight, no hooks needed for demo) ────────────────────────
def generate_gradcam_heatmap(image: Image.Image, predictor, label_idx: int):
    """
    Simplified Grad-CAM visualisation.
    Returns a PIL Image with heatmap overlaid on the X-ray.
    """
    try:
        import torch
        import torch.nn.functional as F
        from torchvision import transforms

        device    = predictor.device
        model     = predictor.model
        transform = predictor.transform

        img_t = transform(image).unsqueeze(0).to(device)

        # Hook to capture gradients and activations from the last conv layer
        activations, gradients = {}, {}

        def fwd_hook(m, inp, out):  activations["val"] = out.detach()
        def bwd_hook(m, gi, go):    gradients["val"]   = go[0].detach()

        # Target the last conv block of ResNet's visual branch
        target_layer = None
        for m in model.visual_branch.modules():
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
        if target_layer is None:
            return None

        fh = target_layer.register_forward_hook(fwd_hook)
        bh = target_layer.register_full_backward_hook(bwd_hook)

        model.zero_grad()
        if hasattr(model, "fusion_head"):
            # Need clinical features — use zeros as neutral
            clin_t = torch.zeros(1, 13).to(device)
            out    = model(img_t, clin_t)
        else:
            out = model(img_t)

        out[0, label_idx].backward()

        fh.remove()
        bh.remove()

        acts  = activations["val"].squeeze()     # (C, H, W)
        grads = gradients["val"].squeeze()       # (C, H, W)
        weights = grads.mean(dim=(1, 2))         # GAP over spatial dims
        cam   = (weights[:, None, None] * acts).sum(0)
        cam   = F.relu(cam)
        cam   = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam_np = cam.cpu().numpy()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        cam_resized = np.array(
            Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
                image.size, Image.BILINEAR
            )
        ) / 255.0

        colormap = cm.jet(cam_resized)[:, :, :3]  # (H, W, 3)
        overlay  = (0.55 * np.array(image) / 255.0 + 0.45 * colormap)
        overlay  = (overlay * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(overlay)

    except Exception:
        return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lungs.png", width=64)
    st.title("Settings")

    models_dir = st.text_input("Models directory", value="./models")

    st.markdown("---")
    st.markdown("**About this system**")
    st.markdown("""
    - ResNet-50 for X-ray analysis
    - Ensemble (RF + GB) for clinical data
    - Early fusion combining both
    - Classes: Normal, Pneumonia, TB, COVID-19
    """)

    st.markdown("---")
    st.markdown("**⚠️ Disclaimer**")
    st.caption(
        "This tool is for academic purposes only. "
        "It is NOT a substitute for professional medical diagnosis."
    )


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🫁 Multimodal Lung Infection Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Fusing chest X-ray images with clinical data for AI-assisted diagnosis</div>', unsafe_allow_html=True)

# Load model
predictor, model_error = load_predictor(models_dir)

if model_error:
    st.markdown(f"""
    <div class="warn-box">
    <b>Models not found.</b> {model_error}<br><br>
    Run the training pipeline first:<br>
    <code>python dataset_prep.py --demo</code><br>
    <code>python train_clinical.py</code><br>
    <code>python train_resnet.py</code><br>
    <code>python fusion_model.py</code>
    </div>
    """, unsafe_allow_html=True)

# ─── Main layout ──────────────────────────────────────────────────────────────
tab_predict, tab_evaluate, tab_about = st.tabs(["🔬 Predict", "📊 Evaluate", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_input, col_result = st.columns([1, 1], gap="large")

    # ── Left: inputs ──────────────────────────────────────────────────────
    with col_input:
        st.subheader("1. Upload Chest X-Ray")
        uploaded_file = st.file_uploader(
            "Drop your X-ray image here",
            type=["jpg", "jpeg", "png"],
            help="JPEG or PNG format, any resolution — we resize to 224×224."
        )

        pil_image = None
        if uploaded_file:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image, caption="Uploaded X-ray", use_column_width=True)

        st.subheader("2. Enter Clinical Data")
        use_clinical = st.toggle(
            "Include clinical parameters (improves accuracy)",
            value=True
        )

        clinical_dict = {}
        if use_clinical:
            with st.expander("📋 Vital Signs & Lab Values", expanded=True):
                for field_id, label, lo, hi, default, step, unit in CLINICAL_FIELDS:
                    val = st.slider(
                        f"{label} ({unit})", min_value=float(lo), max_value=float(hi),
                        value=float(default), step=float(step), key=field_id
                    )
                    clinical_dict[field_id] = val

            with st.expander("🩺 Symptoms", expanded=True):
                cols = st.columns(2)
                for i, (field_id, label) in enumerate(CLINICAL_FLAGS):
                    with cols[i % 2]:
                        clinical_dict[field_id] = int(st.checkbox(label, key=field_id))

        st.subheader("3. Predict")
        predict_btn = st.button(
            "🔍 Analyse & Predict",
            type="primary",
            disabled=(pil_image is None or predictor is None),
            use_container_width=True,
        )

        if pil_image is None:
            st.markdown('<div class="info-box">Upload a chest X-ray image to begin.</div>',
                        unsafe_allow_html=True)
        if predictor is None:
            st.markdown('<div class="warn-box">Models not loaded — train the models first.</div>',
                        unsafe_allow_html=True)

    # ── Right: results ────────────────────────────────────────────────────
    with col_result:
        if predict_btn and pil_image and predictor:
            with st.spinner("Analysing image and clinical data …"):
                result = predictor.predict(
                    image=pil_image,
                    clinical_dict=clinical_dict if use_clinical else None,
                )

            label      = result["label"]
            confidence = result["confidence"]
            probs      = result["probabilities"]
            icon       = CLASS_ICONS[label]
            desc       = CLASS_DESCRIPTIONS[label]

            # ── Prediction card ───────────────────────────────────────────
            st.markdown(f"""
            <div class="prediction-card pred-{label}">
              <div style="font-size:2rem">{icon}</div>
              <div style="font-size:1.6rem;font-weight:700;margin:0.4rem 0">{label}</div>
              <div style="font-size:0.95rem">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence metrics ────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-box">
                  <div class="metric-value">{confidence*100:.1f}%</div>
                  <div class="metric-label">Confidence</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                mode_label = "Fusion" if (use_clinical and predictor.mode == "fusion") else "Image only"
                st.markdown(f"""
                <div class="metric-box">
                  <div class="metric-value" style="font-size:1.2rem">{mode_label}</div>
                  <div class="metric-label">Analysis mode</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                risk = "Low" if label == "NORMAL" else ("Medium" if confidence < 0.75 else "High")
                risk_color = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}[risk]
                st.markdown(f"""
                <div class="metric-box">
                  <div class="metric-value" style="color:{risk_color}">{risk}</div>
                  <div class="metric-label">Risk level</div>
                </div>""", unsafe_allow_html=True)

            # ── Probability bar chart ─────────────────────────────────────
            st.subheader("Class Probabilities")
            fig = go.Figure(go.Bar(
                x=list(probs.keys()),
                y=[v * 100 for v in probs.values()],
                marker_color=[CLASS_COLORS[c] for c in probs.keys()],
                text=[f"{v*100:.1f}%" for v in probs.values()],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis_title="Probability (%)", yaxis_range=[0, 105],
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=10, b=10), height=280,
                font=dict(size=13),
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig, use_container_width=True)

            # ── Grad-CAM heatmap ──────────────────────────────────────────
            st.subheader("Grad-CAM — Attention Map")
            with st.spinner("Generating attention heatmap …"):
                cam_image = generate_gradcam_heatmap(pil_image, predictor, result["label_id"])

            if cam_image:
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(pil_image, caption="Original X-ray", use_column_width=True)
                with img_col2:
                    st.image(cam_image, caption="Regions of interest (Grad-CAM)", use_column_width=True)
                st.caption("🔴 Red/yellow regions indicate the model's focus areas for classification.")
            else:
                st.info("Grad-CAM unavailable in current mode.")

            # ── Downloadable report ───────────────────────────────────────
            st.subheader("Download Report")
            report = {
                "prediction":       label,
                "confidence":       confidence,
                "probabilities":    probs,
                "analysis_mode":    predictor.mode,
                "clinical_data":    clinical_dict if use_clinical else None,
                "disclaimer":       "For academic purposes only. Not a medical diagnosis.",
            }
            st.download_button(
                "⬇️ Download JSON Report",
                data=json.dumps(report, indent=2),
                file_name="lung_infection_report.json",
                mime="application/json",
            )

        elif not predict_btn:
            st.markdown("""
            <div style="text-align:center;padding:4rem 2rem;color:#94a3b8">
              <div style="font-size:4rem">🫁</div>
              <div style="font-size:1.1rem;margin-top:1rem">
                Upload an X-ray and click Analyse to see the prediction
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_evaluate:
    st.subheader("Model Evaluation Dashboard")

    models_path = Path(models_dir)
    clinical_report_path = models_path / "clinical_report.json"
    fusion_history_path  = models_path / "fusion_history.json"
    resnet_history_path  = models_path / "resnet_history.json"

    # Performance metrics table from your report
    st.markdown("#### Performance Summary (from paper)")
    perf_df = pd.DataFrame({
        "Model":       ["ResNet-50 (Image only)", "Clinical Ensemble (BART proxy)", "Early Fusion (Combined)"],
        "Accuracy":    ["96.2%", "91.5%", "99.0%"],
        "Sensitivity": ["94.1%", "88.7%", "97.7%"],
        "Specificity": ["95.3%", "90.2%", "96.8%"],
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    col_ev1, col_ev2 = st.columns(2)

    with col_ev1:
        # Training history plot
        if fusion_history_path.exists():
            with open(fusion_history_path) as f:
                history = json.load(f)
            epochs = list(range(1, len(history["train_acc"]) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=[a*100 for a in history["train_acc"]],
                                     name="Train Acc", line=dict(color="#3b82f6", width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=[a*100 for a in history["val_acc"]],
                                     name="Val Acc", line=dict(color="#22c55e", width=2)))
            fig.update_layout(
                title="Training Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy (%)",
                plot_bgcolor="white", height=300, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train the fusion model to see learning curves here.")

    with col_ev2:
        # Feature importance from clinical model
        if clinical_report_path.exists():
            with open(clinical_report_path) as f:
                cr = json.load(f)
            importance = cr.get("feature_importance", {})
            if importance:
                feats = list(importance.keys())
                vals  = list(importance.values())
                fig = px.bar(
                    x=vals, y=feats, orientation="h",
                    color=vals, color_continuous_scale="Teal",
                    labels={"x": "Importance", "y": "Feature"},
                    title="Clinical Feature Importance",
                )
                fig.update_layout(
                    plot_bgcolor="white", height=300,
                    margin=dict(t=40, b=20), showlegend=False,
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train the clinical model to see feature importance.")

    # Confusion matrix placeholder
    st.markdown("#### Confusion Matrix")
    if clinical_report_path.exists():
        with open(clinical_report_path) as f:
            cr = json.load(f)
        cm = np.array(cr["confusion_matrix"])
        fig = px.imshow(
            cm, text_auto=True,
            x=CLASS_NAMES, y=CLASS_NAMES,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
        )
        fig.update_layout(title="Clinical Model Confusion Matrix", height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        cm_demo = np.array([
            [220, 5,  3,  6],
            [4,  370, 8,  8],
            [2,   6, 820, 22],
            [3,   5,  7,  91],
        ])
        fig = px.imshow(
            cm_demo, text_auto=True,
            x=CLASS_NAMES, y=CLASS_NAMES,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
        )
        fig.update_layout(
            title="Confusion Matrix (demo — train models for real results)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("Project Overview")
    st.markdown("""
    ### Multimodal Approach for Lung Infection Prediction
    **Bannari Amman Institute of Technology — B.Tech IT, 2026**

    **Authors:** Gowtham M (7376232IT506) · Jai Prashanth M (7376232IT507)

    ---

    ### Architecture
    | Component | Technology | Role |
    |-----------|-----------|------|
    | Image encoder | ResNet-50 (pretrained) | Extract 2048-d visual features from chest X-rays |
    | Clinical encoder | Random Forest + Gradient Boosting ensemble | Classify tabular clinical data |
    | Fusion | Early fusion (concatenation + FC layers) | Combine both modalities before classification |
    | Interface | Streamlit | Web UI for clinical use |

    ### Classes Detected
    | Class | Description |
    |-------|-------------|
    | NORMAL | No significant lung pathology |
    | PNEUMONIA | Bacterial or viral pneumonia |
    | TB | Pulmonary tuberculosis |
    | COVID-19 | COVID-19 pneumonia |

    ### Clinical Features Used
    Age, gender, body temperature, oxygen saturation, WBC count, CRP level,
    cough duration, fever, dyspnea, hemoptysis, night sweats, weight loss, smoking status.

    ### Performance (from evaluation)
    - **Fusion model accuracy:** 99.0%
    - **Sensitivity:** 97.7%
    - **Specificity:** 96.8%

    ---
    ⚠️ *For academic/research purposes only. Not for clinical diagnosis.*
    """)
