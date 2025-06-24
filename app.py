import streamlit as st
import os
from PyPDF2 import PdfReader
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import zipfile # Import the zipfile module

st.title("IA Basada en PDFs")
st.write("Ingresa tu pregunta sobre el contenido de los PDFs:")

# Define the path to the zip file containing the PDF files
pdf_zip_path = 'pdfs.zip' # Assuming pdfs.zip is in the root of the repository

# --- AI Logic Integration ---

# 1. Load and read PDF files from the zip archive
@st.cache_resource  # Cache the loaded data to avoid re-processing on each interaction
def load_and_process_pdfs_from_zip(zip_path):
    extracted_text = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List files within the zip archive
            pdf_files_in_zip = [f for f in zip_ref.namelist() if f.endswith('.pdf')]

            for pdf_file in pdf_files_in_zip:
                try:
                    # Open the PDF file from the zip archive
                    with zip_ref.open(pdf_file) as file:
                        reader = PdfReader(file)
                        text = ''
                        for page_num in range(len(reader.pages)):
                            page = reader.pages[page_num]
                            text += page.extract_text()
                        extracted_text.append(text)
                except Exception as e:
                    st.error(f"Error processing {pdf_file} inside zip: {e}") # Use st.error for Streamlit

    except FileNotFoundError:
        st.error(f"Error: The zip file '{zip_path}' was not found.")
        return [] # Return empty list if zip file is not found
    except Exception as e:
        st.error(f"Error opening or reading zip file '{zip_path}': {e}")
        return []

    # 2. Process and chunk the text
    cleaned_and_chunked_text = []
    chunk_size = 500  # Define a suitable chunk size
    for text in extracted_text:
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        chunks = [cleaned_text[i:i+chunk_size] for i in range(0, len(cleaned_text), chunk_size)]
        cleaned_and_chunked_text.extend(chunks)

    st.success(f"Successfully extracted and chunked text from {len(extracted_text)} PDFs into {len(cleaned_and_chunked_text)} chunks.")
    return cleaned_and_chunked_text

# Load and process PDFs from the zip file
cleaned_and_chunked_text = load_and_process_pdfs_from_zip(pdf_zip_path)

# Check if text was extracted before proceeding
if not cleaned_and_chunked_text:
    st.stop() # Stop the app if no text was loaded


# 3. Create and store embeddings
@st.cache_resource # Cache the model and index
def create_embeddings(chunks):
    if not chunks: # Handle case where chunks are empty
        return None, None
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    embeddings_np = np.array(embeddings).astype('float32')
    dimensionality = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimensionality)
    index.add(embeddings_np)
    st.success(f"Created embeddings and FAISS index with {index.ntotal} embeddings.")
    return model, index

model, index = create_embeddings(cleaned_and_chunked_text)

# Check if model and index were created before proceeding
if model is None or index is None:
    st.stop() # Stop the app if embeddings could not be created


# 4. Configure the language model
@st.cache_resource # Cache the language model
def load_language_model():
    try:
        # Using a better model for Question Answering
        llm_model = pipeline("question-answering", model="distilbert/distilbert-base-uncased")
        st.success("Language model loaded successfully.")
        return llm_model
    except Exception as e:
        st.error(f"Error loading language model: {e}")
        return None

language_model = load_language_model()

# Check if language model was loaded before proceeding
if language_model is None:
    st.stop() # Stop the app if language model could not be loaded


# 5. Implement the search and answer function
def ask_pdf_ai(question, model, index, chunks, llm):
    if not question:
        return "Please enter a question."
    if not chunks or model is None or index is None or llm is None:
        return "AI is not fully initialized. Please check logs for errors."

    try:
        question_embedding = model.encode(question)
        question_embedding = np.array([question_embedding]).astype('float32')

        k = 3  # Number of nearest neighbors to retrieve
        distances, indices = index.search(question_embedding, k)
        relevant_chunks = [chunks[i] for i in indices[0]]
        context = "\n".join(relevant_chunks)

        # Prepare input for the question-answering pipeline
        qa_input = {
            'question': question,
            'context': context
        }

        # Use the language model to generate a response
        # For question-answering pipeline, the output is a dictionary
        try:
            response = llm(qa_input) # Pass the dictionary input

            # The response from question-answering pipeline is a dict with keys like 'answer', 'score', 'start', 'end'
            # We are interested in the 'answer'
            clean_response = response.get('answer', "Could not find an answer in the provided text.")

            # Optionally, add some context or confidence score if needed
            # score = response.get('score')
            # clean_response = f"Answer: {answer} (Score: {score:.2f})"


            return clean_response

        except Exception as e:
            st.error(f"Error during language model inference: {e}")
            # Provide a more informative error or fallback
            return f"An error occurred while generating the response. Details: {e}"

    except Exception as e:
        st.error(f"Error during search or embedding: {e}")
        # Provide a more informative error or fallback
        return f"An error occurred while processing your question. Details: {e}"


# --- Streamlit Interface Logic ---

question = st.text_input("Tu pregunta:")
submit_button = st.button("Obtener respuesta")

if submit_button and question:
    with st.spinner("Buscando respuesta..."):
        # Pass the loaded components to the ask_pdf_ai function
        answer = ask_pdf_ai(question, model, index, cleaned_and_chunked_text, language_model)
        st.write("Respuesta:", answer)
elif submit_button and not question:
    st.warning("Por favor, ingresa una pregunta.")
