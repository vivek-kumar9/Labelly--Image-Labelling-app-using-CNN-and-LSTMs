import string
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_caption(caption):
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    return "startseq " + caption + " endseq"

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
    return caption.replace("startseq", "").replace("endseq", "")
