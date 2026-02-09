import streamlit as st
import numpy as np
import pickle

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# Load trained artifacts
# Load trained captioning model
model = load_model("models/caption_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define max caption length (same as training)
MAX_LENGTH = 34   # adjust if your training value differs

# Load CNN encoder (Xception)
# Load Xception encoder (same as training)
base_model = Xception(weights="imagenet", include_top=False, pooling="avg")
encoder = Model(inputs=base_model.input, outputs=base_model.output)

# Image preprocessing function
def extract_features(image):
    image = image.resize((299, 299))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    feature = encoder.predict(image, verbose=0)
    return feature

# Caption generation (greedy decoding)
def generate_caption(model, tokenizer, feature, max_length):
    caption = "startseq"
    
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        
        yhat = model.predict([feature, seq], verbose=0)
        predicted_word = tokenizer.index_word.get(np.argmax(yhat))
        
        if predicted_word is None:
            break
        
        caption += " " + predicted_word
        
        if predicted_word == "endseq":
            break
    
    final_caption = caption.replace("startseq", "").replace("endseq", "").strip()
    return final_caption

# Streamlit UI
st.set_page_config(page_title="Labelly – Image Captioning", layout="centered")

st.title("🖼️ Labelly – Image Captioning App")
st.write("Upload an image and the model will generate a caption.")

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# Run inference
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Generating caption..."):
        feature = extract_features(image)
        caption = generate_caption(model, tokenizer, feature, MAX_LENGTH)
    
    st.success("Caption Generated")
    st.write("📝 **Caption:**")
    st.write(caption)














