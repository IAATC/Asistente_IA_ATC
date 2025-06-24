import streamlit as st
import os
from PyPDF2 import PdfReader
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import zipfile # Import the zipfile module
import torch # Import torch for model loading if needed

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
                        # Limit pages to avoid excessive memory usage for huge PDFs
                        for page_num in range(min(len(reader.pages), 100)): # Process max 100 pages per PDF
                            page = reader.pages[page_num]
                            page_text = page.extract_text()
                            if page_text: # Only add text if extraction was successful
                                text += page_text + "\n" # Add newline for readability between pages
                        if text: # Only add if any text was extracted from the PDF
                             extracted_text.append(text)
                except Exception as e:
                    st.warning(f"Error processing {pdf_file} inside zip: {e}") # Use st.warning instead of error for individual file issues

    except FileNotFoundError:
        st.error(f"Error: The zip file '{zip_path}' was not found.")
        return [] # Return empty list if zip file is not found
    except Exception as e:
        st.error(f"Error opening or reading zip file '{zip_path}': {e}")
        return []

    # 2. Process and chunk the text
    cleaned_and_chunked_text = []
    chunk_size = 500  # Define a suitable chunk size
    chunk_overlap = 50 # Define overlap to maintain context
    for text in extracted_text:
        # Basic cleaning: remove excessive whitespace and newline characters
        cleaned_text = re.sub(r'\s+', ' ', text).strip()

        # Divide text into chunks with overlap
        # Using a simple sliding window for chunking
        if cleaned_text: # Only chunk if there is text
            chunks = []
            for i in range(0, len(cleaned_text), chunk_size - chunk_overlap):
                chunk = cleaned_text[i : i + chunk_size]
                chunks.append(chunk)
            cleaned_and_chunked_text.extend(chunks)


    st.success(f"Successfully extracted and chunked text from {len(extracted_text)} PDFs into {len(cleaned_and_chunked_text)} chunks.")
    return cleaned_and_chunked_text

# Load and process PDFs from the zip file
cleaned_and_chunked_text = load_and_process_pdfs_from_zip(pdf_zip_path)

# Check if text was extracted before proceeding
if not cleaned_and_chunked_text:
    st.error("No text extracted from PDFs. Cannot proceed.")
    st.stop() # Stop the app if no text was loaded


# 3. Create and store embeddings
# Using a smaller, faster model for embeddings
@st.cache_resource # Cache the model and index
def create_embeddings(chunks):
    if not chunks: # Handle case where chunks are empty
        return None, None, None
    try:
        # Using all-MiniLM-L6-v2 as it's efficient
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedding_model.encode(chunks, show_progress_bar=False)
        embeddings_np = np.array(embeddings).astype('float32')
        dimensionality = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimensionality)
        index.add(embeddings_np)
        st.success(f"Created embeddings and FAISS index with {index.ntotal} embeddings.")
        return embedding_model, index, embeddings_np
    except Exception as e:
        st.error(f"Error creating embeddings or FAISS index: {e}")
        return None, None, None


embedding_model, index, embeddings_np = create_embeddings(cleaned_and_chunked_text)

# Check if model and index were created before proceeding
if embedding_model is None or index is None:
    st.error("Could not create embeddings or FAISS index. Cannot proceed.")
    st.stop() # Stop the app if embeddings could not be created


# 4. Configure the language model (Question Answering pipeline)
@st.cache_resource # Cache the language model
def load_language_model():
    try:
        # Using a better model for Question Answering
        # deepset/roberta-base-squad2 is a good option, but might be large
        # You might also consider 'distilbert-base-uncased-distilled-squad' as a smaller alternative
        model_name = "deepset/roberta-base-squad2"
        # Download model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        # Create the question-answering pipeline
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

        st.success(f"Language model '{model_name}' loaded successfully.")
        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading language model '{model_name}': {e}")
        # Provide more specific error for common issues like resource limits
        if "out of memory" in str(e).lower():
             st.error("Model loading failed due to insufficient memory. Consider a smaller model.")
        return None

language_model = load_language_model()

# Check if language model was loaded before proceeding
if language_model is None:
    st.error("Could not load language model. Cannot proceed.")
    st.stop() # Stop the app if language model could not be loaded


# 5. Implement the search and answer function
def ask_pdf_ai(question, embedding_model, index, chunks, qa_pipeline):
    if not question:
        return "Please enter a question."
    if not chunks or embedding_model is None or index is None or qa_pipeline is None:
        return "AI is not fully initialized. Please check logs for errors."

    try:
        # 1. Generate embedding for the question
        question_embedding = embedding_model.encode(question)
        question_embedding = np.array([question_embedding]).astype('float32')

        # 2. Search the FAISS index for the most similar text chunks
        k = 5  # Number of nearest neighbors to retrieve (increased for better context)
        distances, indices = index.search(question_embedding, k)

        # 3. Retrieve the actual text chunks
        relevant_chunks = [chunks[i] for i in indices[0]]
        context = "\n".join(relevant_chunks)

        # 4. Prepare input for the question-answering pipeline
        qa_input = {
            'question': question,
            'context': context
        }

        # 5. Use the question-answering pipeline to generate a response
        try:
            # Pass the dictionary input to the pipeline
            response = qa_pipeline(qa_input)

            # The response from question-answering pipeline is a dict with keys like 'answer', 'score', 'start', 'end'
            # We are interested in the 'answer'
            answer = response.get('answer', "Could not find a specific answer in the provided text chunks.")
            score = response.get('score', 0.0) # Get confidence score

            # You can add a confidence threshold if you only want to show answers above a certain score
            # if score < 0.5: # Example threshold
            #     return "Could not find a confident answer in the provided text chunks."

            # Return the answer, maybe with the score for debugging/info
            return f"{answer}" # Or f"Respuesta: {answer} (Confianza: {score:.2f})"

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
        answer = ask_pdf_ai(question, embedding_model, index, cleaned_and_chunked_text, language_model)
        st.write("Respuesta:", answer)
elif submit_button and not question:
    st.warning("Por favor, ingresa una pregunta.")
