import os
import string
import pickle
import numpy as np

from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add
from tensorflow.keras.optimizers import Adam

from nltk.translate.bleu_score import corpus_bleu

# Load Captions (Flickr8k.token.txt)
def load_captions(filename):
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            image_id = tokens[0].split('.')[0]
            caption = ' '.join(tokens[1:])
            mapping.setdefault(image_id, []).append(caption)
    return mapping

# Clean Captions (NLP preprocessing)
def clean_captions(mapping):
    table = str.maketrans('', '', string.punctuation)
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower()
            caption = caption.translate(table)
            caption = caption.replace('[^a-z]', '')
            caption = 'startseq ' + caption + ' endseq'
            captions[i] = caption

# Prepare Tokenizer & Vocabulary
caption_file = "data/Flickr8k_text/Flickr8k.token.txt"
captions = load_captions(caption_file)
clean_captions(captions)

all_captions = [c for caps in captions.values() for c in caps]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in all_captions)

# save tokenizer for inference
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Image Feature Extraction (Xception Encoder)
base_model = Xception(weights="imagenet", include_top=False, pooling="avg")
encoder = Model(inputs=base_model.input, outputs=base_model.output)

features = {}

image_dir = "data/Flickr8k_Dataset"

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    image = load_img(img_path, target_size=(299, 299))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    image_id = img_name.split('.')[0]
    features[image_id] = encoder.predict(image)

with open("features/image_features.pkl", "wb") as f:
    pickle.dump(features, f)

# Data Generator (Teacher Forcing)
def data_generator(captions, features, tokenizer, max_length):
    while True:
        for image_id, caps in captions.items():
            feature = features[image_id][0]
            for caption in caps:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq = pad_sequences([seq[:i]], maxlen=max_length)
                    out_seq = seq[i]
                    yield ([feature, in_seq], out_seq)

# Model Architecture (CNN + LSTM)
image_input = Input(shape=(2048,))
image_dense = Dense(256, activation="relu")(image_input)

text_input = Input(shape=(max_length,))
text_embedding = Embedding(vocab_size, 256, mask_zero=True)(text_input)
text_lstm = LSTM(256)(text_embedding)

decoder = Add()([image_dense, text_lstm])
output = Dense(vocab_size, activation="softmax")(decoder)

model = Model(inputs=[image_input, text_input], outputs=output)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam()
)

model.summary()

# Training Loop
steps = sum(len(caps) for caps in captions.values())
generator = data_generator(captions, features, tokenizer, max_length)

model.fit(
    generator,
    epochs=20,
    steps_per_epoch=steps,
    verbose=1
)

model.save("models/caption_model.h5")

# BLEU Score Evaluation
def generate_caption(model, tokenizer, feature, max_length):
    caption = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([feature, seq], verbose=0)
        word = tokenizer.index_word.get(np.argmax(yhat))
        if word is None:
            break
        caption += " " + word
        if word == "endseq":
            break
    return caption.split()[1:-1]


actual, predicted = [], []

for image_id, caps in captions.items():
    y_pred = generate_caption(model, tokenizer, features[image_id], max_length)
    actual.append([c.split()[1:-1] for c in caps])
    predicted.append(y_pred)

print("BLEU-1:", corpus_bleu(actual, predicted, weights=(1, 0, 0, 0)))















