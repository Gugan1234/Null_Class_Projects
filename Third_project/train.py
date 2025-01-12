import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load your JSON file
with open(r"C:/Users/gugan/OneDrive/Desktop/Null_class_Project3/cs_ds_papers.json", "r") as file:
    data = json.load(file)

# Prepare data structures
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# Process the JSON data
for entry in data:
    abstract = entry["abstract"]
    title = entry["title"]
    
    # Tokenize and process the abstract text
    word_list = nltk.word_tokenize(abstract)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list if word not in ignore_letters and word not in stop_words]
    words.extend(word_list)
    documents.append((word_list, title))
    if title not in classes:
        classes.append(title)

# Lemmatize and remove duplicates
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes for future use
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Prepare the training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    # Create a bag of words
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Create the output row
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# Shuffle and convert to NumPy array
random.shuffle(training)
training = np.array(training)

# Split into features and labels
train_X = training[:, : len(words)]
train_Y = training[:, len(words) :]

# Build the improved model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=(len(train_X[0]),), activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(train_Y[0]), activation="softmax"),
])

# Compile the model with Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the model with early stopping and learning rate scheduling
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * (0.1 ** (epoch // 50)))

hist = model.fit(
    np.array(train_X), 
    np.array(train_Y), 
    epochs=400, 
    batch_size=5, 
    callbacks=[callback, lr_schedule], 
    verbose=1
)

# Save the model
model.save("chatbot_model.keras", hist)
print("Model training complete and saved!")