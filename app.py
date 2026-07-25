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
# Session History Storage
# -----------------------------------

if "history" not in st.session_state:

    st.session_state.history = []



# -----------------------------------
# Title
# -----------------------------------

st.title(
    "Counterfeit 1000 BDT Banknote Detection"
)


st.write(
    "CNN-based counterfeit detection using EfficientNetB0"
)


st.divider()



# -----------------------------------
# Input Selection
# -----------------------------------

option = st.radio(
    "Choose input method:",
    [
        "Upload Image",
        "Use Camera"
    ]
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


    image = Image.open(
        uploaded_file
    )


    st.image(
        image,
        caption="Uploaded Banknote",
        use_container_width=True
    )



    img = image.convert(
        "RGB"
    )


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



        st.subheader(
            "🔍 Detection Result"
        )



        # According to your trained model:
        # 0 = Fake
        # 1 = Real


        if probability >= threshold:


            

            result = "Real"

            confidence = probability * 100


            st.success(
                "✅ Genuine Banknote (REAL)"
            )


        else:


            

            result = "Fake"

            confidence = (1 - probability) * 100


            st.error(
                "❌ Counterfeit Banknote (FAKE)"
            )



        st.write(
            "### Confidence Score"
        )


        st.progress(
            int(confidence)
        )


        st.write(
            f"**Confidence:** {confidence:.2f}%"
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
        # Heatmap Improvement
        # -----------------------------------

        # Better normalization
        heatmap = np.maximum(
            heatmap,
            0
        )


        max_value = np.max(
            heatmap
        )


        if max_value != 0:

            heatmap = heatmap / max_value


        # Noise reduction by removing weak activation
        heatmap = np.clip(
            heatmap,
            0.20,
            1.0
        )


        # Convert to 0-255 format
        heatmap = np.uint8(
            255 * heatmap
        )


        # Cleaner Grad-CAM visualization
        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_TURBO
        )


        # Original image
        original_img = np.array(
            image.convert("RGB")
        )


        # Overlay
        superimposed_img = cv2.addWeighted(
            original_img,
            0.8,
            heatmap,
            0.2,
            0
        )


        st.subheader(
            "🧠 CNN Attention Visualization (Grad-CAM)"
        )


        st.image(
            superimposed_img,
            caption="Highlighted regions influencing CNN prediction",
            use_container_width=True
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

st.divider()



title_col, delete_col = st.columns(
    [6,1]
)



with title_col:

    st.subheader(
        "📋Detection History"
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



    # Table Header

    h1,h2,h3,h4,h5,h6 = st.columns(
        [1,3,1,2,2,1]
    )


    h1.write(
        "No."
    )


    h2.write(
        "Image"
    )


    h3.write(
        "Result"
    )


    h4.write(
        "Confidence"
    )


    h5.write(
        "Date Time"
    )


    h6.write(
        "Action"
    )



    st.divider()



    for index,row in history_df.iterrows():



        c1,c2,c3,c4,c5,c6 = st.columns(
            [1,3,1,2,2,1]
        )



        c1.write(
            index + 1
        )



        c2.write(
            row["image_name"]
        )



        c3.write(
            row["result"]
        )



        c4.write(
            f'{row["confidence"]:.2f}%'
        )



        c5.write(
            row["date_time"]
        )



        if c6.button(
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


    st.info(
        "No detection history available"
    )





# -----------------------------------
# Footer
# -----------------------------------

st.divider()



st.caption(
    "Capstone Project | CNN-Based Approach for Detecting Counterfeit 1000 BDT Banknotes Using Watermark Analysis"
)