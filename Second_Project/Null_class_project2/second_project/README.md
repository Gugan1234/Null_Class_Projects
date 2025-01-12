#2. Implement a system for dynamically expanding the chatbot's knowledge base. Create a mechanism to periodically update the vector database with new information from specified sources. Expected Outcome: A chatbot that can automatically incorporate new information into its responses over time.

# myapp.py
- This file uses webscrapping method to fetch the information form the site and uses CAG operation (CACHE AUGMENTED GENERATION)
# app.py
- This file uses the directory document to fetch the information and uses FAISS to restore the information that learns in certain period  of time and it uses the RAG in this process

# Working Video
Link: https://www.linkedin.com/posts/gugan-r-92b4b2261_cag-rag-nullclass-activity-7284114983569760256-vNcm?utm_source=share&utm_medium=member_desktop
