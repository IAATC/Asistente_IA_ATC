import streamlit as st
import os
from PyPDF2 import PdfReader
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.title("IA Basada en PDFs")
st.write("Ingresa tu pregunta sobre el contenido de los PDFs:")

# Define the directory containing the PDF files
pdf_directory = '/content'

# --- AI Logic Integration ---

# 1. Load and read PDF files (re-using the previous code)
@st.cache_resource  # Cache the loaded data to avoid re-processing on each interaction
def load_and_process_pdfs(directory):
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    extracted_text = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory, pdf_file)
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ''
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text()
                extracted_text.append(text)
        except Exception as e:
            st.error(f"Error processing {pdf_file}: {e}") # Use st.error for Streamlit

    # 2. Process and chunk the text (re-using the previous code)
    cleaned_and_chunked_text = []
    chunk_size = 500  # Define a suitable chunk size
    for text in extracted_text:
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        chunks = [cleaned_text[i:i+chunk_size] for i in range(0, len(cleaned_text), chunk_size)]
        cleaned_and_chunked_text.extend(chunks)

    return cleaned_and_chunked_text

# Load and process PDFs
cleaned_and_chunked_text = load_and_process_pdfs(pdf_directory)

# 3. Create and store embeddings (re-using the previous code)
@st.cache_resource # Cache the model and index
def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    embeddings_np = np.array(embeddings).astype('float32')
    dimensionality = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimensionality)
    index.add(embeddings_np)
    return model, index

model, index = create_embeddings(cleaned_and_chunked_text)

# 4. Configure the language model (re-using the previous code)
@st.cache_resource # Cache the language model
def load_language_model():
    # Using a smaller model for demonstration, consider a larger one for better results
    return pipeline("text-generation", model="distilgpt2")

language_model = load_language_model()


# 5. Implement the search and answer function (re-using the previous code)
def ask_pdf_ai(question, model, index, chunks, llm):
    question_embedding = model.encode(question)
    question_embedding = np.array([question_embedding]).astype('float32')

    k = 3  # Number of nearest neighbors to retrieve
    distances, indices = index.search(question_embedding, k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n".join(relevant_chunks)

    prompt = f"Based on the following text:\n{context}\n\nAnswer the question: {question}"

    # Adjust max_new_tokens as needed
    response = llm(prompt, max_new_tokens=200, num_return_sequences=1)[0]['generated_text']

    # Attempt to clean the response (this might need refinement)
    try:
        # Find the start of the answer by looking for the end of the prompt
        # This is a simple approach and might not work perfectly with all LLMs
        prompt_end_indicator = f"Answer the question: {question}"
        prompt_end_index = response.find(prompt_end_indicator)
        if prompt_end_index != -1:
            # Add the length of the indicator and a little extra to get past it
            clean_response = response[prompt_end_index + len(prompt_end_indicator):].strip()
            # Remove potential remnants of the prompt or unwanted characters at the beginning
            clean_response = re.sub(r'^["\'\s,.:;]+', '', clean_response)
        else:
            # If prompt indicator not found, return the full response
            clean_response = response.strip()
    except Exception as e:
        st.warning(f"Could not fully clean response: {e}")
        clean_response = response.strip() # Fallback to returning the full response


    return clean_response


# --- Streamlit Interface Logic ---

question = st.text_input("Tu pregunta:")
submit_button = st.button("Obtener respuesta")

if submit_button and question:
    with st.spinner("Buscando respuesta..."):
        answer = ask_pdf_ai(question, model, index, cleaned_and_chunked_text, language_model)
        st.write("Respuesta:", answer)
elif submit_button and not question:
    st.warning("Por favor, ingresa una pregunta.")