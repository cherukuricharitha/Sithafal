from bs4 import BeautifulSoup
import requests
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

file_path = "faiss_store_openai.pkl"

# Function to scrape content from a website
def scrape_website(url):
    # Send a request to the website
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from all paragraph tags (you can adjust this as needed)
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    else:
        print(f"Failed to retrieve the webpage: {url}")
        return ""

# Process websites after input
def process_websites(urls):
    all_text = ""

    # Scrape text from all websites
    for url in urls:
        print(f"Processing website: {url}")
        extracted_text = scrape_website(url)
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

# User input for URLs
urls = input("Enter the website URLs (comma-separated): ").split(',')

# Run processing after URL input
process_websites(urls)

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
