# # AIzaSyAx3QXRZuJ2ARpzi6REq8HcCX4xolp4Gjc - gemini Api key
# # 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
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

import streamlit as st
import os
import google.generativeai as genai

# Set up the API key for Gemini AI
os.environ["GOOGLE_API_KEY"] = "AIzaSyAx3QXRZuJ2ARpzi6REq8HcCX4xolp4Gjc"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Function to get the Gemini AI response
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Initialize Streamlit app
st.set_page_config(page_title="GEMINI CHATBOT DEMO")

st.header("Gemini Application")

# Input field for the user to ask questions
input_text = st.text_input("Input: ", key="input")

# Button to send the query
submit = st.button("Click here to send")

# Display chat messages with emojis
if submit:
    if input_text:
        # User message
        st.chat_message("user", avatar="🧑‍💻").write(input_text)

        # Get Gemini AI response
        response = get_gemini_response(input_text)

        # Display bot response with emoji
        st.subheader("The Response is")
        for chunk in response:
            st.chat_message("assistant", avatar="🤖").write(chunk.text)

        # Optionally, display the full chat history
        st.write(chat.history)
    else:
        st.write("Please enter some text.")
