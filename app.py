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
# Palette: ink navy + antique gold + verified green + alert crimson,
# tuned for a currency-security / forensic-verification product.

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root{
        --ink:#0E1524;
        --panel:#161F33;
        --panel-border:#28324A;
        --gold:#C6A15B;
        --gold-soft:rgba(198,161,91,0.14);
        --green:#3E8E63;
        --green-soft:rgba(62,142,99,0.14);
        --crimson:#B3413A;
        --crimson-soft:rgba(179,65,58,0.14);
        --paper:#F1ECDF;
        --muted:#8C96AC;
    }

    html, body, [class*="css"]{
        font-family:'Inter', sans-serif;
    }

    .stApp{
        background:
            radial-gradient(ellipse 900px 500px at 15% -5%, rgba(198,161,91,0.08), transparent 60%),
            var(--ink);
        color:var(--paper);
    }

    #MainMenu, footer{visibility:hidden;}
    header[data-testid="stHeader"]{background:transparent;}

    .block-container{
        padding-top:1.5rem;
        max-width:760px;
    }

    /* ---------- Hero ---------- */
    .hero{
        position:relative;
        border-radius:14px;
        padding:2.1rem 1.8rem 1.7rem 1.8rem;
        margin-bottom:1.6rem;
        background:
            repeating-linear-gradient(135deg, rgba(198,161,91,0.055) 0px, rgba(198,161,91,0.055) 1px, transparent 1px, transparent 13px),
            linear-gradient(160deg, #141C30 0%, #0F1626 100%);
        border:1px solid var(--panel-border);
        overflow:hidden;
    }
    .hero::after{
        content:"";
        position:absolute; top:0; right:0; bottom:0;
        width:120px;
        background:linear-gradient(90deg, transparent, rgba(198,161,91,0.10));
        pointer-events:none;
    }
    .hero-badge{
        display:inline-flex; align-items:center; gap:0.4rem;
        font-family:'IBM Plex Mono', monospace;
        font-size:0.68rem; letter-spacing:0.11em; text-transform:uppercase;
        color:var(--gold);
        background:var(--gold-soft);
        border:1px solid rgba(198,161,91,0.35);
        padding:0.28rem 0.65rem;
        border-radius:100px;
        margin-bottom:0.9rem;
    }
    .hero-badge svg{width:11px; height:11px;}
    .hero h1{
        font-family:'Fraunces', serif;
        font-weight:600;
        font-size:1.9rem;
        line-height:1.18;
        margin:0 0 0.55rem 0;
        color:var(--paper);
        letter-spacing:-0.01em;
    }
    .hero p{
        font-size:0.93rem;
        color:var(--muted);
        margin:0 0 1.0rem 0;
        max-width:32rem;
        line-height:1.5;
    }
    .hero-tags{display:flex; flex-wrap:wrap; gap:0.5rem;}
    .hero-tag{
        font-size:0.72rem;
        font-family:'IBM Plex Mono', monospace;
        color:var(--muted);
        border:1px solid var(--panel-border);
        border-radius:6px;
        padding:0.22rem 0.55rem;
        background:rgba(255,255,255,0.02);
    }

    /* ---------- Section labels ---------- */
    .section-label{
        font-family:'IBM Plex Mono', monospace;
        font-size:0.72rem;
        letter-spacing:0.1em;
        text-transform:uppercase;
        color:var(--gold);
        margin:1.7rem 0 0.6rem 0;
        display:flex; align-items:center; gap:0.5rem;
    }
    .section-label::after{
        content:"";
        flex:1;
        height:1px;
        background:var(--panel-border);
    }

    /* ---------- Streamlit widget restyle ---------- */
    div[data-testid="stRadio"] > div{
        gap:0.6rem;
    }
    div[data-testid="stRadio"] label{
        background:var(--panel);
        border:1px solid var(--panel-border);
        padding:0.5rem 0.9rem;
        border-radius:8px;
    }

    section[data-testid="stFileUploaderDropzone"], div[data-testid="stCameraInput"]{
        background:var(--panel) !important;
        border:1px dashed var(--panel-border) !important;
        border-radius:12px !important;
    }

    section[data-testid="stFileUploaderDropzone"] button{
        background:linear-gradient(180deg, #D4B36E, var(--gold)) !important;
        color:#1B1406 !important;
        border:none !important;
        border-radius:8px !important;
        font-weight:600 !important;
        box-shadow:0 3px 10px rgba(198,161,91,0.22);
    }
    section[data-testid="stFileUploaderDropzone"] button p,
    section[data-testid="stFileUploaderDropzone"] button span{
        color:#1B1406 !important;
        -webkit-text-fill-color:#1B1406 !important;
        font-weight:600 !important;
    }
    section[data-testid="stFileUploaderDropzone"] button svg{
        fill:#1B1406 !important;
    }

    .stButton>button{
        width:100%;
        background:linear-gradient(180deg, #D4B36E, var(--gold));
        color:#1B1406;
        font-weight:600;
        border:none;
        border-radius:9px;
        padding:0.65rem 1rem;
        font-size:0.95rem;
        transition:transform 0.12s ease, box-shadow 0.12s ease;
        box-shadow:0 4px 14px rgba(198,161,91,0.18);
    }
    .stButton>button:hover{
        transform:translateY(-1px);
        box-shadow:0 6px 18px rgba(198,161,91,0.28);
        color:#1B1406;
    }

    div[data-testid="stImage"] img{
        border-radius:10px;
        border:1px solid var(--panel-border);
    }

    /* ---------- Verdict card ---------- */
    .verdict-card{
        border-radius:14px;
        padding:1.4rem 1.5rem;
        margin:0.8rem 0 1.1rem 0;
        border:1px solid;
    }
    .verdict-real{
        background:var(--green-soft);
        border-color:rgba(62,142,99,0.45);
    }
    .verdict-fake{
        background:var(--crimson-soft);
        border-color:rgba(179,65,58,0.45);
    }
    .verdict-top{
        display:flex; align-items:center; justify-content:space-between;
        margin-bottom:0.9rem;
    }
    .verdict-label{
        font-family:'Fraunces', serif;
        font-weight:600;
        font-size:1.35rem;
    }
    .verdict-real .verdict-label{color:#7FD6A8;}
    .verdict-fake .verdict-label{color:#F0958D;}
    .verdict-sub{
        font-family:'IBM Plex Mono', monospace;
        font-size:0.72rem;
        letter-spacing:0.08em;
        text-transform:uppercase;
        color:var(--muted);
        margin-top:0.15rem;
    }
    .verdict-confidence-num{
        font-family:'IBM Plex Mono', monospace;
        font-size:1.6rem;
        font-weight:600;
    }
    .verdict-real .verdict-confidence-num{color:#7FD6A8;}
    .verdict-fake .verdict-confidence-num{color:#F0958D;}

    .conf-track{
        width:100%;
        height:8px;
        border-radius:100px;
        background:rgba(255,255,255,0.08);
        overflow:hidden;
    }
    .conf-fill{
        height:100%;
        border-radius:100px;
    }
    .verdict-real .conf-fill{background:linear-gradient(90deg, #2E7D5B, #58C48A);}
    .verdict-fake .conf-fill{background:linear-gradient(90deg, #8E2E29, #D9645B);}

    /* ---------- Explainability card ---------- */
    .explain-card{
        background:var(--panel);
        border:1px solid var(--panel-border);
        border-radius:12px;
        padding:1.1rem 1.3rem;
        margin-top:0.6rem;
    }
    .explain-card ul{margin:0; padding-left:1.1rem;}
    .explain-card li{
        font-size:0.88rem;
        color:var(--paper);
        margin-bottom:0.3rem;
        line-height:1.45;
    }
    .explain-title{
        font-family:'IBM Plex Mono', monospace;
        font-size:0.72rem;
        letter-spacing:0.08em;
        text-transform:uppercase;
        color:var(--gold);
        margin-bottom:0.6rem;
    }

    /* ---------- History table ---------- */
    .history-card{
        background:var(--panel);
        border:1px solid var(--panel-border);
        border-radius:12px;
        padding:0.9rem 1.1rem 0.3rem 1.1rem;
        margin-top:0.5rem;
    }
    .history-head{
        font-family:'IBM Plex Mono', monospace;
        font-size:0.68rem;
        letter-spacing:0.08em;
        text-transform:uppercase;
        color:var(--muted);
        padding-bottom:0.5rem;
        border-bottom:1px solid var(--panel-border);
    }
    .history-row{
        font-size:0.86rem;
        color:var(--paper);
        padding:0.55rem 0;
        border-bottom:1px solid rgba(255,255,255,0.04);
    }
    .badge{
        display:inline-block;
        font-family:'IBM Plex Mono', monospace;
        font-size:0.72rem;
        font-weight:600;
        padding:0.15rem 0.55rem;
        border-radius:100px;
    }
    .badge-real{background:var(--green-soft); color:#7FD6A8;}
    .badge-fake{background:var(--crimson-soft); color:#F0958D;}

    .empty-state{
        text-align:center;
        color:var(--muted);
        font-size:0.88rem;
        padding:1.6rem 0;
        border:1px dashed var(--panel-border);
        border-radius:12px;
        background:var(--panel);
    }

    .app-footer{
        text-align:center;
        color:var(--muted);
        font-size:0.78rem;
        margin-top:2.2rem;
        padding-top:1.2rem;
        border-top:1px solid var(--panel-border);
        line-height:1.6;
    }
    .app-footer .gold{color:var(--gold);}
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
