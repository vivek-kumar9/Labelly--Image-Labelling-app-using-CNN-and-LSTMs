# Labelly - Image Captioning App (CNN + LSTM)

## 📌 Overview
Labelly is an image captioning application that generates natural language descriptions for images using a CNN–LSTM encoder–decoder architecture.  
The project combines computer vision and natural language processing to demonstrate multimodal deep learning in an end-to-end pipeline.

The model is trained on the Flickr8k dataset and deployed using Streamlit for interactive inference.

---

## 🧠 Architecture

Image → Xception CNN → Image Feature Vector  
Text → Tokenizer → Embedding → LSTM Decoder → Caption

### Key Design Choices
- Encoder: Xception (pretrained on ImageNet)
- Decoder: LSTM
- Training: Teacher forcing
- Inference: Greedy decoding
- Evaluation Metric: BLEU score

---

## 📂 Dataset
- Dataset: Flickr8k
- Images: 8,000
- Captions: 5 per image (40,000 total)

The dataset is not included in this repository due to size and licensing constraints.

---

## ⚙️ Tech Stack
- Python
- TensorFlow / Keras
- CNN (Xception)
- LSTM
- NLP (tokenization, sequence modeling)
- Streamlit
- NLTK (BLEU evaluation)

---

## 🔄 Workflow
1. Caption preprocessing (cleaning, tokenization, start/end tokens)
2. Image feature extraction using pretrained CNN
3. Sequence modeling using LSTM
4. Training with teacher forcing
5. Evaluation using BLEU score
6. Deployment via Streamlit app

---

## 📊 Results
- BLEU-1 score: > 0.5 on Flickr8k
- Generated captions are grammatically coherent and semantically aligned with image content

BLEU-1 was chosen due to dataset size and to avoid over-claiming performance.

---

## 🖥️ Streamlit App
The application allows users to:
- Upload an image
- Extract visual features using the trained CNN encoder
- Generate captions using the trained LSTM decoder

This demonstrates how a deep learning model can be wrapped into a simple interactive application.

---

## ⚠️ Limitations
- Trained on a relatively small dataset
- Captions may be generic
- No attention mechanism
- Greedy decoding instead of beam search

This project is intended as a learning and demonstration system.

---

## 🚀 Future Improvements
- Add attention mechanism
- Use beam search decoding
- Train on larger datasets (e.g., MS-COCO)
- Improve caption diversity

---

## 🎯 One-Line Summary
Built a CNN–LSTM based image captioning system using Xception and LSTM, trained on Flickr8k, achieving a BLEU-1 score above 0.5 and deployed via Streamlit.
