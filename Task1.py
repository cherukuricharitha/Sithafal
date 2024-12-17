!pip install pdfminer.six
!pip install streamlit
!pip install pickle5
!pip install langchain
!pip install langchain-groq
!pip install faiss-cpu
!pip install huggingface_hub
!pip install -U langchain-community
from pdfminer.high_level import extract_text
import os
import pickle
import time
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Initialize LLM
llm = ChatGroq(temperature=0, groq_api_key="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", model_name="llama-3.1-70b-versatile")

# File upload in Colab
from google.colab import files
uploaded_files = files.upload()

file_path = "faiss_store_openai.pkl"

# Process PDFs after upload
def process_pdfs():
    all_text = ""

    # Extract text from all PDFs
    for uploaded_file in uploaded_files.keys():
        extracted_text = extract_text(uploaded_file)
        all_text += extracted_text + "\n"

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(all_text)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_openai = FAISS.from_texts(text_chunks, embeddings)

    # Save FAISS index
    print("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

    print("Text extracted and FAISS index saved.")

# Run processing after file upload
process_pdfs()

# Query input
query = input("Ask a Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        # Get response
        result = chain.run(query)

        # Display answer
        print("Answer:")
        print(result)
