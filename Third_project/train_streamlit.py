import streamlit as st
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import nltk
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import pickle

# Download required NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define model path and check if it exists
MODEL_PATH = "chatbot_model.keras"
WORDS_PATH = "words.pkl"
CLASSES_PATH = "classes.pkl"

# Check TensorFlow version for compatibility
st.write(f"TensorFlow version: {tf.__version__}")

# Load the model with error handling
def load_model():
    try:
        # Attempt to load the model without compiling (skip optimizer and training state)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.write("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the trained model and other necessary files
model = load_model()

# Check if model is loaded before proceeding
if model is None:
    st.stop()

try:
    words = pickle.load(open(WORDS_PATH, "rb"))
    classes = pickle.load(open(CLASSES_PATH, "rb"))
except Exception as e:
    st.error(f"Error loading words or classes: {e}")
    st.stop()

# Debug: print classes and words


# Initialize Ollama LLaMA model using Langchain for interactive responses
llama_model = Ollama(model="llama3")  # Assuming "llama-3" is the name of the model

# Create a prompt template for LLaMA interaction
prompt_template = """
You are a helpful assistant who answers questions about computer science research papers.

User Query: {query}
Research Area: {topic}

Response:
"""
prompt = PromptTemplate(input_variables=["query", "topic"], template=prompt_template)
chain = LLMChain(llm=llama_model, prompt=prompt)

# Define a function to preprocess input text
def preprocess_input(text):
    tokens = nltk.word_tokenize(text)
    st.write(f"Tokens: {tokens}")  # Debugging line
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    st.write(f"Lemmatized Tokens: {tokens}")  # Debugging line
    bag = [1 if word in tokens else 0 for word in words]
    return np.array(bag)

# Predict the class of the input text
def predict_class(text):
    bag = preprocess_input(text)
    prediction = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.2  # Increased threshold for more confidence
    results = [[i, res] for i, res in enumerate(prediction) if res > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Main Streamlit app
def main():
    st.title("Interactive Research Paper Chatbot")
    st.write("Chat with a model trained on CS research papers!")
    
    # Sidebar for topic selection
    topic_options = ["Machine Learning", "Computer Vision", "Natural Language Processing", "Reinforcement Learning", "Optimization", "Other"]
    selected_topic = st.sidebar.selectbox("Select a Research Area", topic_options)
    
    st.write(f"You have selected: {selected_topic}")
    
    user_input = st.text_input("You:", placeholder="Ask about research papers, categories, or topics...")

    if st.button("Submit"):
        if user_input.strip():
            # First, classify the user's input to predict its topic
            predicted_classes = predict_class(user_input)
            if predicted_classes:
                # Get the class with the highest confidence
                class_id = predicted_classes[0][0]
                predicted_class = classes[class_id]
                
                # Use LLaMA for enhanced response generation
                response = chain.run(query=user_input, topic=selected_topic)
                st.write(f"Bot: {response}")
            else:
                # If no class with enough confidence
                st.write("Bot: I'm not sure about that. Can you rephrase or try another question?")
        else:
            st.warning("Please enter a message to chat!")

if __name__ == "__main__":
    main()
