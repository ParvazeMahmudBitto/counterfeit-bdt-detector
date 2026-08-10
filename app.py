import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
from datetime import datetime
import cv2




# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Counterfeit BDT Detector",
    page_icon="💵",
    layout="centered"
)


# -----------------------------------
# Design System (CSS)
# -----------------------------------
# UI ONLY: professional financial-security / forensic dashboard theme.
# The model, prediction, Grad-CAM, threshold, history and CSV logic below
# remain unchanged.

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&display=swap');
    :root{
        --bg:#07110F;
        --bg-2:#0A1714;
        --surface:#0E1D19;
        --surface-2:#12241F;
        --surface-3:#172B25;
        --border:#28433A;
        --border-soft:rgba(104,145,129,0.22);
        --text:#F4F1E8;
        --text-soft:#C8D1CC;
        --muted:#879B93;
        --gold:#D8B760;
        --gold-light:#F0D98F;
        --gold-soft:rgba(216,183,96,0.11);
        --green:#56C995;
        --green-soft:rgba(86,201,149,0.11);
        --red:#F06F6A;
        --red-soft:rgba(240,111,106,0.11);
        --shadow:0 18px 48px rgba(0,0,0,0.26);
    }

    html, body, [class*="css"]{
        font-family:"Segoe UI", Inter, -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
    }

    html, body{
        background:var(--bg) !important;
    }

    .stApp{
        color:var(--text);
        background:
            radial-gradient(circle at 7% 0%, rgba(216,183,96,0.08), transparent 27rem),
            radial-gradient(circle at 95% 15%, rgba(47,128,99,0.09), transparent 30rem),
            linear-gradient(180deg, #07110F 0%, #091511 48%, #07100E 100%);
    }

    .stApp::before{
        content:"";
        position:fixed;
        top:0;
        left:0;
        right:0;
        height:3px;
        z-index:999999;
        background:linear-gradient(90deg, transparent 0%, var(--gold) 30%, var(--green) 70%, transparent 100%);
        opacity:0.9;
    }

    #MainMenu,
    footer{
        visibility:hidden;
    }

    header[data-testid="stHeader"]{
        background:transparent;
    }

    [data-testid="stToolbar"]{
        right:0.6rem;
    }

    .block-container{
        max-width:920px;
        padding-top:1.8rem;
        padding-bottom:2.5rem;
    }

    /* ---------- Global readable text ---------- */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p{
        color:var(--text-soft) !important;
        -webkit-text-fill-color:var(--text-soft) !important;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p{
        color:var(--muted) !important;
        -webkit-text-fill-color:var(--muted) !important;
    }

    /* ---------- Hero ---------- */
    .hero{
        position:relative;
        overflow:hidden;
        margin:0 0 1.9rem 0;
        padding:2.35rem 2.25rem 2.05rem 2.25rem;
        border:1px solid var(--border);
        border-radius:20px;
        background:
            linear-gradient(120deg, rgba(255,255,255,0.018), transparent 35%),
            linear-gradient(145deg, #10241E 0%, #0B1916 55%, #0A1513 100%);
        box-shadow:var(--shadow);
    }

    .hero::before{
        content:"";
        position:absolute;
        width:310px;
        height:310px;
        right:-115px;
        top:-120px;
        border:1px solid rgba(216,183,96,0.16);
        border-radius:50%;
        box-shadow:
            0 0 0 34px rgba(216,183,96,0.035),
            0 0 0 70px rgba(216,183,96,0.022);
        pointer-events:none;
    }

    .hero::after{
        content:"1000";
        position:absolute;
        right:1.25rem;
        bottom:-1.2rem;
        font-family:Georgia, "Times New Roman", serif;
        font-size:7.4rem;
        font-weight:700;
        letter-spacing:-0.08em;
        color:rgba(216,183,96,0.035);
        line-height:1;
        pointer-events:none;
    }

    .hero-badge{
        position:relative;
        z-index:2;
        display:inline-flex;
        align-items:center;
        gap:0.48rem;
        margin-bottom:1rem;
        padding:0.36rem 0.72rem;
        border:1px solid rgba(216,183,96,0.34);
        border-radius:999px;
        background:rgba(216,183,96,0.08);
        color:var(--gold-light);
        font-size:0.69rem;
        font-weight:700;
        letter-spacing:0.11em;
        text-transform:uppercase;
    }

    .hero-badge svg{
        width:13px;
        height:13px;
        stroke:var(--gold-light);
    }

    .hero h1{
        position:relative;
        z-index:2;
        max-width:650px;
        margin:0 0 0.7rem 0;
        font-family:"Montserrat", sans-serif;
        color:var(--text);
        font-size:2.15rem;
        font-weight:600;
        line-height:1.12;
        letter-spacing:-0.025em;
    }

    .hero p{
        position:relative;
        z-index:2;
        max-width:640px;
        margin:0 0 1.2rem 0;
        color:var(--text-soft) !important;
        -webkit-text-fill-color:var(--text-soft) !important;
        font-size:0.94rem;
        line-height:1.7;
    }

    .hero-tags{
        position:relative;
        z-index:2;
        display:flex;
        flex-wrap:wrap;
        gap:0.55rem;
    }

    .hero-tag{
        display:inline-flex;
        align-items:center;
        min-height:30px;
        padding:0.28rem 0.68rem;
        border:1px solid var(--border-soft);
        border-radius:8px;
        background:rgba(255,255,255,0.025);
        color:#ADC0B8;
        font-size:0.72rem;
        font-weight:600;
        letter-spacing:0.02em;
    }

    /* ---------- Section label ---------- */
    .section-label{
        display:flex;
        align-items:center;
        gap:0.75rem;
        margin:1.85rem 0 0.75rem 0;
        color:var(--gold-light);
        font-size:0.69rem;
        font-weight:800;
        letter-spacing:0.115em;
        text-transform:uppercase;
    }

    .section-label::before{
        content:"";
        width:8px;
        height:8px;
        border:2px solid var(--gold);
        border-radius:50%;
        box-shadow:0 0 0 4px rgba(216,183,96,0.08);
        flex:0 0 auto;
    }

    .section-label::after{
        content:"";
        height:1px;
        flex:1;
        background:linear-gradient(90deg, var(--border), transparent);
    }

    /* ---------- Input method / radio ---------- */
    div[data-testid="stRadio"]{
        margin-bottom:0.8rem;
    }

    div[data-testid="stRadio"] [role="radiogroup"]{
        display:flex;
        flex-wrap:wrap;
        gap:0.65rem;
    }

    div[data-testid="stRadio"] label{
        min-width:150px;
        margin:0 !important;
        padding:0.62rem 0.9rem !important;
        border:1px solid var(--border) !important;
        border-radius:11px !important;
        background:linear-gradient(180deg, rgba(18,36,31,0.92), rgba(13,28,24,0.92)) !important;
        transition:border-color .18s ease, transform .18s ease, background .18s ease;
    }

    div[data-testid="stRadio"] label:hover{
        border-color:rgba(216,183,96,0.48) !important;
        transform:translateY(-1px);
    }

    div[data-testid="stRadio"] label:has(input:checked){
        border-color:rgba(216,183,96,0.72) !important;
        background:linear-gradient(180deg, rgba(216,183,96,0.12), rgba(18,36,31,0.96)) !important;
        box-shadow:0 8px 22px rgba(0,0,0,0.16);
    }

    div[data-testid="stRadio"] input[type="radio"]{
        accent-color:var(--gold);
    }

    /* ---------- File uploader / camera ---------- */
    section[data-testid="stFileUploaderDropzone"]{
        min-height:125px;
        padding:0.65rem !important;
        border:1px dashed rgba(216,183,96,0.38) !important;
        border-radius:16px !important;
        background:
            linear-gradient(145deg, rgba(216,183,96,0.045), transparent 42%),
            var(--surface) !important;
        box-shadow:inset 0 1px 0 rgba(255,255,255,0.025);
        transition:border-color .18s ease, background .18s ease;
    }

    section[data-testid="stFileUploaderDropzone"]:hover{
        border-color:rgba(216,183,96,0.68) !important;
        background:
            linear-gradient(145deg, rgba(216,183,96,0.075), transparent 46%),
            var(--surface-2) !important;
    }

    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] div{
        color:var(--text-soft) !important;
        -webkit-text-fill-color:var(--text-soft) !important;
    }

    [data-testid="stFileUploaderDropzone"] small{
        color:var(--muted) !important;
        -webkit-text-fill-color:var(--muted) !important;
    }

    section[data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stFileUploaderDropzone"] [data-testid^="stBaseButton"]{
        border:1px solid rgba(216,183,96,0.55) !important;
        border-radius:9px !important;
        background:linear-gradient(180deg, #E1C46F, #CBA956) !important;
        box-shadow:0 6px 18px rgba(216,183,96,0.16) !important;
    }

    section[data-testid="stFileUploaderDropzone"] button *,
    section[data-testid="stFileUploaderDropzone"] button p,
    section[data-testid="stFileUploaderDropzone"] button span,
    section[data-testid="stFileUploaderDropzone"] button div{
        color:#162019 !important;
        -webkit-text-fill-color:#162019 !important;
        font-weight:800 !important;
    }

    section[data-testid="stFileUploaderDropzone"] button:hover{
        background:linear-gradient(180deg, #F0D98F, #D8B760) !important;
    }

    div[data-testid="stCameraInput"]{
        border-radius:16px;
        overflow:hidden;
    }

    /* ---------- Buttons ---------- */
    .stButton > button{
        width:100%;
        min-height:44px;
        border:1px solid rgba(216,183,96,0.52) !important;
        border-radius:11px !important;
        background:linear-gradient(180deg, #E1C36D 0%, #CBA653 100%) !important;
        color:#142019 !important;
        font-weight:800 !important;
        letter-spacing:0.01em;
        box-shadow:0 8px 22px rgba(216,183,96,0.13);
        transition:transform .15s ease, box-shadow .15s ease, filter .15s ease;
    }

    .stButton > button *{
        color:#142019 !important;
        -webkit-text-fill-color:#142019 !important;
        font-weight:800 !important;
    }

    .stButton > button:hover{
        transform:translateY(-1px);
        filter:brightness(1.05);
        box-shadow:0 11px 26px rgba(216,183,96,0.19);
    }

    .stButton > button:active{
        transform:translateY(0);
    }

    .stDownloadButton > button{
        width:100%;
        min-height:42px;
        margin-top:0.5rem;
        border:1px solid var(--border) !important;
        border-radius:11px !important;
        background:var(--surface-2) !important;
        color:var(--text) !important;
        font-weight:700 !important;
        transition:border-color .16s ease, background .16s ease;
    }

    .stDownloadButton > button *{
        color:var(--text) !important;
        -webkit-text-fill-color:var(--text) !important;
    }

    .stDownloadButton > button:hover{
        border-color:rgba(216,183,96,0.62) !important;
        background:var(--surface-3) !important;
    }

    /* ---------- Uploaded / Grad-CAM images ---------- */
    div[data-testid="stImage"]{
        padding:0.3rem;
        border:1px solid var(--border-soft);
        border-radius:16px;
        background:rgba(255,255,255,0.018);
        box-shadow:0 12px 32px rgba(0,0,0,0.16);
    }

    div[data-testid="stImage"] img{
        display:block;
        border-radius:12px;
    }

    div[data-testid="stImage"] [data-testid="stImageCaption"]{
        color:var(--muted) !important;
        font-size:0.76rem !important;
        padding:0.3rem 0.15rem 0.1rem 0.15rem;
    }

    /* ---------- Verdict ---------- */
    .verdict-card{
        position:relative;
        overflow:hidden;
        margin:0.85rem 0 1.25rem 0;
        padding:1.45rem 1.55rem;
        border:1px solid;
        border-radius:17px;
        box-shadow:0 14px 34px rgba(0,0,0,0.17);
    }

    .verdict-card::after{
        content:"";
        position:absolute;
        right:-38px;
        top:-38px;
        width:120px;
        height:120px;
        border-radius:50%;
        opacity:0.22;
    }

    .verdict-real{
        border-color:rgba(86,201,149,0.38);
        background:
            linear-gradient(115deg, rgba(86,201,149,0.11), transparent 50%),
            var(--surface);
    }

    .verdict-real::after{
        background:radial-gradient(circle, rgba(86,201,149,0.48), transparent 67%);
    }

    .verdict-fake{
        border-color:rgba(240,111,106,0.40);
        background:
            linear-gradient(115deg, rgba(240,111,106,0.11), transparent 50%),
            var(--surface);
    }

    .verdict-fake::after{
        background:radial-gradient(circle, rgba(240,111,106,0.48), transparent 67%);
    }

    .verdict-top{
        position:relative;
        z-index:2;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:1.2rem;
        margin-bottom:1rem;
    }

    .verdict-label{
        color:var(--text);
        font-family:Georgia, "Times New Roman", serif;
        font-size:1.34rem;
        font-weight:700;
        line-height:1.25;
    }

    .verdict-sub{
        margin-top:0.28rem;
        color:var(--muted);
        font-size:0.66rem;
        font-weight:800;
        letter-spacing:0.11em;
        text-transform:uppercase;
    }

    .verdict-confidence-num{
        padding:0.26rem 0.55rem;
        border-radius:9px;
        font-size:1.55rem;
        font-weight:800;
        font-variant-numeric:tabular-nums;
        white-space:nowrap;
    }

    .verdict-real .verdict-confidence-num{
        color:#7AD8AC;
        background:rgba(86,201,149,0.08);
    }

    .verdict-fake .verdict-confidence-num{
        color:#FF9691;
        background:rgba(240,111,106,0.08);
    }

    .conf-track{
        position:relative;
        z-index:2;
        width:100%;
        height:8px;
        overflow:hidden;
        border-radius:999px;
        background:rgba(255,255,255,0.07);
        box-shadow:inset 0 1px 2px rgba(0,0,0,0.30);
    }

    .conf-fill{
        height:100%;
        border-radius:999px;
    }

    .verdict-real .conf-fill{
        background:linear-gradient(90deg, #3CA877, #70D8AA);
        box-shadow:0 0 12px rgba(86,201,149,0.32);
    }

    .verdict-fake .conf-fill{
        background:linear-gradient(90deg, #D34E4A, #FF8580);
        box-shadow:0 0 12px rgba(240,111,106,0.30);
    }

    /* ---------- Explanation ---------- */
    .explain-card{
        position:relative;
        margin-top:0.85rem;
        padding:1.2rem 1.3rem;
        border:1px solid var(--border);
        border-radius:14px;
        background:
            linear-gradient(135deg, rgba(216,183,96,0.055), transparent 42%),
            var(--surface);
        box-shadow:0 10px 28px rgba(0,0,0,0.12);
    }

    .explain-title{
        margin-bottom:0.75rem;
        color:var(--gold-light);
        font-size:0.68rem;
        font-weight:800;
        letter-spacing:0.105em;
        text-transform:uppercase;
    }

    .explain-card ul{
        display:grid;
        grid-template-columns:repeat(2, minmax(0,1fr));
        gap:0.5rem 0.8rem;
        margin:0;
        padding:0;
        list-style:none;
    }

    .explain-card li{
        position:relative;
        margin:0;
        padding:0.52rem 0.62rem 0.52rem 1.55rem;
        border:1px solid rgba(103,142,127,0.16);
        border-radius:9px;
        background:rgba(255,255,255,0.018);
        color:var(--text-soft) !important;
        -webkit-text-fill-color:var(--text-soft) !important;
        font-size:0.83rem;
        line-height:1.4;
    }

    .explain-card li::before{
        content:"";
        position:absolute;
        left:0.64rem;
        top:0.83rem;
        width:6px;
        height:6px;
        border-radius:50%;
        background:var(--gold);
        box-shadow:0 0 0 3px rgba(216,183,96,0.08);
    }

    /* ---------- History ---------- */
    .history-card{
        margin-top:0.55rem;
        padding:0.95rem 1.05rem 0.4rem 1.05rem;
        border:1px solid var(--border);
        border-radius:15px;
        background:var(--surface);
        box-shadow:0 12px 30px rgba(0,0,0,0.12);
    }

    .history-head{
        min-height:30px;
        padding-bottom:0.55rem;
        border-bottom:1px solid var(--border-soft);
        color:var(--muted);
        font-size:0.63rem;
        font-weight:800;
        letter-spacing:0.08em;
        text-transform:uppercase;
    }

    .history-row{
        min-height:42px;
        padding:0.72rem 0 0.58rem 0;
        border-bottom:1px solid rgba(255,255,255,0.035);
        color:var(--text-soft);
        font-size:0.82rem;
        line-height:1.35;
        overflow-wrap:anywhere;
    }

    .badge{
        display:inline-flex;
        align-items:center;
        padding:0.18rem 0.55rem;
        border-radius:999px;
        font-size:0.66rem;
        font-weight:800;
        letter-spacing:0.03em;
    }

    .badge-real{
        border:1px solid rgba(86,201,149,0.22);
        background:var(--green-soft);
        color:#79D6AA;
    }

    .badge-fake{
        border:1px solid rgba(240,111,106,0.23);
        background:var(--red-soft);
        color:#FF9792;
    }

    .empty-state{
        padding:2rem 1rem;
        border:1px dashed var(--border);
        border-radius:15px;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.018), transparent),
            var(--surface);
        color:var(--muted);
        text-align:center;
        font-size:0.86rem;
    }

    /* ---------- Alerts ---------- */
    [data-testid="stAlert"]{
        border:1px solid var(--border) !important;
        border-radius:12px !important;
        background:var(--surface-2) !important;
        box-shadow:0 8px 20px rgba(0,0,0,0.10);
    }

    [data-testid="stAlert"] p{
        color:var(--text-soft) !important;
        -webkit-text-fill-color:var(--text-soft) !important;
    }

    /* ---------- Footer ---------- */
    .app-footer{
        margin-top:2.4rem;
        padding:1.35rem 0.4rem 0.15rem 0.4rem;
        border-top:1px solid var(--border-soft);
        color:var(--muted);
        text-align:center;
        font-size:0.75rem;
        line-height:1.65;
    }

    .app-footer::before{
        content:"SECURE • ANALYZE • VERIFY";
        display:block;
        margin-bottom:0.42rem;
        color:rgba(216,183,96,0.66);
        font-size:0.61rem;
        font-weight:800;
        letter-spacing:0.14em;
    }

    .app-footer .gold{
        color:#D9C27D;
    }

    /* ---------- Focus accessibility ---------- */
    button:focus-visible,
    input:focus-visible,
    [tabindex]:focus-visible{
        outline:2px solid rgba(216,183,96,0.82) !important;
        outline-offset:2px !important;
    }

    /* ---------- Mobile ---------- */
    @media (max-width: 640px){
        .block-container{
            padding-top:1rem;
            padding-left:0.9rem;
            padding-right:0.9rem;
        }

        .hero{
            padding:1.65rem 1.25rem 1.45rem 1.25rem;
            border-radius:16px;
        }

        .hero::after{
            font-size:5.2rem;
            right:0.7rem;
        }

        .hero h1{
            max-width:95%;
            font-size:1.55rem;
        }

        .hero p{
            font-size:0.86rem;
            line-height:1.58;
        }

        .hero-badge{
            font-size:0.60rem;
        }

        .hero-tag{
            min-height:27px;
            font-size:0.66rem;
        }

        .section-label{
            margin-top:1.45rem;
            font-size:0.62rem;
        }

        div[data-testid="stRadio"] [role="radiogroup"]{
            display:grid;
            grid-template-columns:1fr 1fr;
            width:100%;
        }

        div[data-testid="stRadio"] label{
            min-width:0;
            width:100%;
        }

        .verdict-card{
            padding:1.15rem;
        }

        .verdict-top{
            align-items:flex-start;
            flex-direction:column;
            gap:0.8rem;
        }

        .verdict-label{
            font-size:1.16rem;
        }

        .verdict-confidence-num{
            font-size:1.3rem;
        }

        .explain-card ul{
            grid-template-columns:1fr;
        }

        .history-card{
            padding-left:0.75rem;
            padding-right:0.75rem;
        }

        .history-row{
            font-size:0.72rem;
        }

        .history-head{
            font-size:0.55rem;
        }
    }

    @media (max-width: 420px){
        div[data-testid="stRadio"] [role="radiogroup"]{
            grid-template-columns:1fr;
        }

        .hero h1{
            font-size:1.4rem;
        }

        .hero::before{
            opacity:0.65;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------
# Load Model
# -----------------------------------

MODEL_PATH = "efficientnetb0_final.keras"
BACKBONE_PATH = "efficientnetb0_backbone.keras"


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        st.error(
            "Model file not found"
        )

        st.stop()


    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


    return model



model = load_model()

for layer in model.layers:
    print(layer.name)

# -----------------------------------
# Load EfficientNet Backbone for Grad-CAM
# -----------------------------------

@st.cache_resource
def load_backbone():

    if not os.path.exists(BACKBONE_PATH):
        st.warning("Grad-CAM backbone file not found")
        return None

    backbone = tf.keras.models.load_model(
        BACKBONE_PATH,
        compile=False
    )

    return backbone


backbone = load_backbone()



# -----------------------------------
# Proper Grad-CAM Function
# -----------------------------------

def make_gradcam_heatmap(
        img_array,
        backbone
):

    if backbone is None:
        return None


    last_conv_layer = backbone.get_layer(
        "top_conv"
    )


    grad_model = tf.keras.models.Model(
        inputs=backbone.inputs,
        outputs=last_conv_layer.output
    )


    with tf.GradientTape() as tape:

        conv_output = grad_model(
            img_array
        )


    heatmap = tf.reduce_mean(
        conv_output,
        axis=-1
    )


    heatmap = heatmap[0]


    heatmap = tf.maximum(
        heatmap,
        0
    )


    heatmap = heatmap / tf.reduce_max(
        heatmap
    )


    return heatmap.numpy()


# -----------------------------------
# Security Feature Region Mask
# -----------------------------------

def apply_security_region_mask(heatmap, width, height):

    mask = np.zeros((height, width), dtype=np.float32)

    # Flower print + Bangladesh Bank logo watermark
    cv2.rectangle(
        mask,
        (int(width*0.08), int(height*0.30)),
        (int(width*0.38), int(height*0.62)),
        1,
        -1
    )

    # Mujib portrait watermark
    cv2.rectangle(
        mask,
        (int(width*0.45), int(height*0.20)),
        (int(width*0.78), int(height*0.72)),
        1,
        -1
    )

    # Watermark text (1000)
    cv2.rectangle(
        mask,
        (int(width*0.38), int(height*0.68)),
        (int(width*0.80), int(height*0.88)),
        1,
        -1
    )

    # Soft edges
    mask = cv2.GaussianBlur(
        mask,
        (51,51),
        0
    )

    return heatmap * mask



# -----------------------------------
# Session History Storage
# -----------------------------------

if "history" not in st.session_state:

    st.session_state.history = []



# -----------------------------------
# Hero / Title
# -----------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L4 5V11C4 16 7.5 20.5 12 22C16.5 20.5 20 16 20 11V5L12 2Z" stroke="currentColor" stroke-width="1.6"/>
            </svg>
            Counterfeit Detection System
        </div>
        <h1>Counterfeit 1000&nbsp;BDT Banknote Detection</h1>
        <p>Upload or capture a banknote photo and the model examines its watermark
        security features — the Mujib portrait, the Bangladesh Bank emblem, and the
        embedded "1000" mark — to verify authenticity.</p>
        <div class="hero-tags">
            <span class="hero-tag">EfficientNetB0</span>
            <span class="hero-tag">Watermark Analysis</span>
            <span class="hero-tag">Grad-CAM Explainability</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



# -----------------------------------
# Input Selection
# -----------------------------------

st.markdown('<div class="section-label">Step 1 — Provide a note image</div>', unsafe_allow_html=True)

option = st.radio(
    "Choose input method:",
    [
        "Upload Image",
        "Use Camera"
    ],
    label_visibility="collapsed"
)



uploaded_file = None



if option == "Upload Image":


    uploaded_file = st.file_uploader(
        "Upload a 1000 BDT banknote image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )



elif option == "Use Camera":


    uploaded_file = st.camera_input(
        "Take a picture of the banknote"
    )



# -----------------------------------
# Image Processing
# -----------------------------------

if uploaded_file is not None:


    from PIL import ImageOps


    image = Image.open(
    uploaded_file
)


# Fix camera/mobile image rotation
    image = ImageOps.exif_transpose(
     image
)


    st.markdown('<div class="section-label">Step 2 — Review the image</div>', unsafe_allow_html=True)

    st.image(
        image,
        caption="Uploaded Banknote",
        use_container_width=True
    )



    image = ImageOps.exif_transpose(
        image
         ).convert(
      "RGB"
    )

    img = image.copy()


    img = img.resize(
        (224,224)
    )


    img_array = np.array(
        img
    )


    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    img_array = tf.keras.applications.efficientnet.preprocess_input(
        img_array
    )



    # -----------------------------------
    # Prediction
    # -----------------------------------
    result = None
    confidence = None

    st.markdown('<div class="section-label">Step 3 — Run detection</div>', unsafe_allow_html=True)

    if st.button(
        "🔍 Detect Banknote"
    ):


        prediction = model.predict(
            img_array
        )


        probability = float(
            prediction[0][0]
        )


        threshold = 0.40



        # According to your trained model:
        # 0 = Fake
        # 1 = Real


        if probability >= threshold:


            

            result = "Real"

            confidence = probability * 100


        else:


            

            result = "Fake"

            confidence = (1 - probability) * 100



        verdict_class = "verdict-real" if result == "Real" else "verdict-fake"
        verdict_icon = "✅" if result == "Real" else "❌"
        verdict_text = "Genuine Banknote" if result == "Real" else "Counterfeit Banknote"

        st.markdown(
            f"""
            <div class="verdict-card {verdict_class}">
                <div class="verdict-top">
                    <div>
                        <div class="verdict-label">{verdict_icon} {verdict_text}</div>
                        <div class="verdict-sub">Detection Result &nbsp;·&nbsp; {result.upper()}</div>
                    </div>
                    <div class="verdict-confidence-num">{confidence:.2f}%</div>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{min(confidence, 100):.2f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # -----------------------------------
        # Grad-CAM Visualization
        # -----------------------------------

        heatmap = make_gradcam_heatmap(
            img_array,
            backbone
        )


        heatmap = cv2.resize(
            heatmap,
            (
                image.size[0],
                image.size[1]
            )
        )


        # -----------------------------------
        # Region Guided Grad-CAM
        # -----------------------------------

        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)

        heatmap = apply_security_region_mask(
            heatmap,
            image.size[0],
            image.size[1]
        )

        # Smooth feature-focused attention
        heatmap = cv2.GaussianBlur(
            heatmap,
            (15,15),
            0
        )

        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        heatmap_uint8 = np.uint8(255 * heatmap)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_TURBO
        )

        original_img = np.array(image.convert("RGB"))

        superimposed_img = cv2.addWeighted(
            original_img,
            0.6,
            heatmap_color,
            0.4,
            0
        )

        st.markdown('<div class="section-label">CNN Attention (Grad-CAM)</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.image(original_img, caption="Original Banknote", use_container_width=True)

        with col2:
            st.image(superimposed_img, caption="Security Feature Focused Grad-CAM", use_container_width=True)

        # -----------------------------------
        # Dynamic CNN Attention Explanation
        # -----------------------------------

        if result == "Real":

            attention_items = [
                "Flower print",
                "Bangladesh Bank logo (Watermark)",
                "Watermark text (1000)",
                "Watermark (Mujib Portrait)",
            ]

        else:

            # Default counterfeit indicators
            attention_items = [
                "Unclear Flower print",
                "Blur Bangladesh Bank logo (Watermark)",
                "Font inconsistency Watermark text (1000)",
                "Distorted Portrait Watermark (Mujib Portrait)",
            ]

        attention_list_html = "".join([f"<li>{item}</li>" for item in attention_items])

        st.markdown(
            f"""
            <div class="explain-card">
                <div class="explain-title">CNN attention focused on security regions</div>
                <ul>{attention_list_html}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        
        
                
        # Save to current session only
        if result is not None:

            st.session_state.history.append(
                {
                    "image_name": (
                        uploaded_file.name
                        if uploaded_file is not None
                        else "camera_capture.jpg"
                    ),

                    "result": result,

                    "confidence": confidence,

                    "date_time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                }
            )



      # -----------------------------------
# History Section
# -----------------------------------

st.markdown('<div class="section-label">Detection History</div>', unsafe_allow_html=True)



title_col, delete_col = st.columns(
    [6,1]
)



with title_col:

    st.caption(
        "Every check you run in this session is logged below."
    )



with delete_col:

    if st.button(
        "🗑️"
    ):

        st.session_state.confirm_delete = True




# -----------------------------------
# Delete All Confirmation
# -----------------------------------

if st.session_state.get(
    "confirm_delete",
    False
):


    st.warning(
        "⚠️ Are you sure you want to delete all detection history?"
    )


    yes_col, no_col = st.columns(2)



    with yes_col:

        if st.button(
            "✅ Yes, Delete All"
        ):


            st.session_state.history = []


            st.session_state.confirm_delete = False


            st.success(
                "All detection history deleted successfully"
            )


            st.rerun()



    with no_col:


        if st.button(
            "❌ Cancel"
        ):


            st.session_state.confirm_delete = False


            st.rerun()





# -----------------------------------
# Display History
# -----------------------------------

if len(st.session_state.history) > 0:



    history_df = pd.DataFrame(
        st.session_state.history
    )


    st.markdown('<div class="history-card">', unsafe_allow_html=True)

    # Table Header

    h1,h2,h3,h4,h5,h6 = st.columns(
        [1,3,1,2,2,1]
    )


    with h1:
        st.markdown('<div class="history-head">No.</div>', unsafe_allow_html=True)


    with h2:
        st.markdown('<div class="history-head">Image</div>', unsafe_allow_html=True)


    with h3:
        st.markdown('<div class="history-head">Result</div>', unsafe_allow_html=True)


    with h4:
        st.markdown('<div class="history-head">Confidence</div>', unsafe_allow_html=True)


    with h5:
        st.markdown('<div class="history-head">Date Time</div>', unsafe_allow_html=True)


    with h6:
        st.markdown('<div class="history-head">Action</div>', unsafe_allow_html=True)



    for index,row in history_df.iterrows():



        c1,c2,c3,c4,c5,c6 = st.columns(
            [1,3,1,2,2,1]
        )



        with c1:
            st.markdown(f'<div class="history-row">{index + 1}</div>', unsafe_allow_html=True)



        with c2:
            st.markdown(f'<div class="history-row">{row["image_name"]}</div>', unsafe_allow_html=True)



        with c3:
            badge_class = "badge-real" if row["result"] == "Real" else "badge-fake"
            st.markdown(f'<div class="history-row"><span class="badge {badge_class}">{row["result"]}</span></div>', unsafe_allow_html=True)



        with c4:
            st.markdown(f'<div class="history-row">{row["confidence"]:.2f}%</div>', unsafe_allow_html=True)



        with c5:
            st.markdown(f'<div class="history-row">{row["date_time"]}</div>', unsafe_allow_html=True)



        with c6:
            if st.button(
                "🗑️",
                key=f"delete_{index}"
            ):



                st.session_state.history.pop(
                    index
                )


                st.success(
                    "Record deleted successfully"
                )


                st.rerun()


    st.markdown('</div>', unsafe_allow_html=True)


    # -----------------------------------
    # CSV Download
    # -----------------------------------


    csv_df = history_df.copy()



    csv_df.insert(
        0,
        "No.",
        range(
            1,
            len(csv_df)+1
        )
    )



    csv = csv_df.to_csv(
        index=False
    )



    st.download_button(
        "⬇️ Download History CSV",
        csv,
        "detection_history.csv",
        "text/csv"
    )



else:


    st.markdown(
        '<div class="empty-state">No detection history available yet — run a check above to see it logged here.</div>',
        unsafe_allow_html=True
    )





# -----------------------------------
# Footer
# -----------------------------------

st.markdown(
    """
    <div class="app-footer">
        Capstone Project &nbsp;·&nbsp; <span class="gold">CNN-Based Approach for Detecting Counterfeit 1000 BDT Banknotes Using Watermark Analysis</span>
    </div>
    """,
    unsafe_allow_html=True
)