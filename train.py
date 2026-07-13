import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine
data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)

# Combine title + text
data["content"] = data["title"] + " " + data["text"]

# Drop null
data = data.dropna()

X = data["text"]
y = data["label"]

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(sequences, maxlen=100)

y = np.array(y)

# Model
model = Sequential()
model.add(Embedding(5000, 64, input_length=100))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X_pad, y, epochs=3, batch_size=32)

# Save model
model.save("model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model and tokenizer saved!")

print(fake.columns)
print(true.columns)
