import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import ollama
from langchain_community.llms import Ollama

# Load environment variables for API keys
load_dotenv()

# Set up API key for Gemini AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="AIzaSyAx3QXRZuJ2ARpzi6REq8HcCX4xolp4Gjc")

# Initialize Gemini Model
gemini_model = genai.GenerativeModel("gemini-pro")
gemini_chat = gemini_model.start_chat(history=[])

# Initialize Ollama Models
ollama_phi_model = Ollama(model="phi")  # Phi model
ollama_model = Ollama(model="llama3")  # Llama3 model

# Function to get response from Gemini AI
def get_gemini_response(question):
    response = gemini_chat.send_message(question, stream=True)
    return "".join(chunk.text for chunk in response)

# Function to get response from Ollama Phi
def get_ollama_phi_response(question):
    response = ollama_phi_model.stream(input=question)
    return "".join(token for token in response)

# Function to get response from Ollama
def get_ollama_response(question):
    response = ollama_model.stream(input=question)
    return "".join(token for token in response)

# Suggest Best Response function
def suggest_best_response():
    gemini_response = st.session_state["gemini_messages"][-1]["content"]
    ollama_phi_response = st.session_state["ollama_phi_messages"][-1]["content"]
    ollama_response = st.session_state["ollama_messages"][-1]["content"]
    
    responses = {
        "Gemini": len(gemini_response),
        "Phi (Ollama)": len(ollama_phi_response),
        "Ollama": len(ollama_response)
    }
    best_bot = max(responses, key=responses.get)
    return best_bot

# Streamlit app configuration
st.set_page_config(page_title="Chatbot Comparison")

st.title("Chatbot Comparison: Gemini, Phi (Ollama), and Ollama")

# Top-left corner: Suggest Best Response button
if st.button("Suggest Best Response"):
    best_bot = suggest_best_response()
    st.subheader(f"Suggested Best Response: {best_bot}")

# Create three columns for chatbots
col1, col2, col3 = st.columns(3)

# Initialize session state for each chatbot
if "gemini_messages" not in st.session_state:
    st.session_state["gemini_messages"] = [{"role": "assistant", "content": "Hello, how can I assist you today?"}]

if "ollama_phi_messages" not in st.session_state:
    st.session_state["ollama_phi_messages"] = [{"role": "assistant", "content": "Hello, I am Phi from Ollama!"}]

if "ollama_messages" not in st.session_state:
    st.session_state["ollama_messages"] = [{"role": "assistant", "content": "I am Ollama chatbot."}]

# Define and display Gemini Chatbot
with col1:
    st.header("Gemini AI Chatbot")
    for msg in st.session_state["gemini_messages"]:
        st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖").write(msg["content"])

# Define and display Ollama Phi Chatbot
with col2:
    st.header("Phi (Ollama) Chatbot")
    for msg in st.session_state["ollama_phi_messages"]:
        st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖").write(msg["content"])

# Define and display Ollama Chatbot
with col3:
    st.header("Ollama Chatbot")
    for msg in st.session_state["ollama_messages"]:
        st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖").write(msg["content"])

# Shared user input for all chatbots
st.divider()
shared_input = st.text_input("Send message to all chatbots:", key="shared_input")
if st.button("Send to All"):
    if shared_input:
        # Add input to all chat histories
        st.session_state["gemini_messages"].append({"role": "user", "content": shared_input})
        st.session_state["ollama_phi_messages"].append({"role": "user", "content": shared_input})
        st.session_state["ollama_messages"].append({"role": "user", "content": shared_input})
        
        # Get responses
        gemini_response = get_gemini_response(shared_input)
        ollama_phi_response = get_ollama_phi_response(shared_input)
        ollama_response = get_ollama_response(shared_input)

        # Add responses to histories
        st.session_state["gemini_messages"].append({"role": "assistant", "content": gemini_response})
        st.session_state["ollama_phi_messages"].append({"role": "assistant", "content": ollama_phi_response})
        st.session_state["ollama_messages"].append({"role": "assistant", "content": ollama_response})

# Individual user inputs for each chatbot
st.divider()
col_input1, col_input2, col_input3 = st.columns(3)

with col_input1:
    individual_gemini_input = st.text_input("Gemini Input:", key="individual_gemini_input")
    if st.button("Send to Gemini"):
        if individual_gemini_input:
            st.session_state["gemini_messages"].append({"role": "user", "content": individual_gemini_input})
            response = get_gemini_response(individual_gemini_input)
            st.session_state["gemini_messages"].append({"role": "assistant", "content": response})

with col_input2:
    individual_phi_input = st.text_input("Phi Input:", key="individual_phi_input")
    if st.button("Send to Phi"):
        if individual_phi_input:
            st.session_state["ollama_phi_messages"].append({"role": "user", "content": individual_phi_input})
            response = get_ollama_phi_response(individual_phi_input)
            st.session_state["ollama_phi_messages"].append({"role": "assistant", "content": response})

with col_input3:
    individual_ollama_input = st.text_input("Ollama Input:", key="individual_ollama_input")
    if st.button("Send to Ollama"):
        if individual_ollama_input:
            st.session_state["ollama_messages"].append({"role": "user", "content": individual_ollama_input})
            response = get_ollama_response(individual_ollama_input)
            st.session_state["ollama_messages"].append({"role": "assistant", "content": response})
