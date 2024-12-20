import os
import re
from typing import List, Dict
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformersEmbeddings


class WebsiteScraper:

    def _init_(self, url_list: List[str]):
        self.url_list = url_list

    def crawl_and_extract(self):
        extracted_texts = []
        for url in self.url_list:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text()
                extracted_texts.append(text)
            except Exception as e:
                print(f"Error scraping {url}: {e}")
        return extracted_texts

    def segment_text(self, extracted_texts):
        return [text.split("\n") for text in extracted_texts]  

    def generate_embeddings(self, text_chunks):
        embedding_model = SentenceTransformersEmbeddings(model_name="all-MiniLM-L6-v2")
        embeddings = embedding_model.embed_documents(text_chunks)
        return embeddings

    def store_embeddings(self, embeddings):
        faiss_index = FAISS.from_documents(embeddings)
        faiss_index.save_local("faiss_embeddings_db")
        return faiss_index

class QueryHandler:

    def _init_(self, embeddings_store):
        self.embeddings_store = embeddings_store

    def convert_query_to_embeddings(self, query: str):
        embedding_model = SentenceTransformersEmbeddings(model_name="all-MiniLM-L6-v2")
        query_embedding = embedding_model.embed_query(query)  
        return query_embedding

    def perform_similarity_search(self, query_embedding):
        return self.embeddings_store.similarity_search(query_embedding, k=3)

    def retrieve_chunks(self, similar_chunks):
        return similar_chunks

class ResponseGenerator:

    def _init_(self):
        self.llm = pipeline("text-generation", model="gpt-3.5-turbo")  

    def generate_response(self, retrieved_chunks: List[str]):
        context = " ".join(retrieved_chunks)
        response = self.llm(context, max_length=150)  
        return response[0]['generated_text']
def main():
    urls = ["https://example.com", "https://another-example.com"]
    scraper = WebsiteScraper(urls)
    extracted_texts = scraper.crawl_and_extract()
    text_chunks = scraper.segment_text(extracted_texts)
    embeddings = scraper.generate_embeddings(text_chunks)
    faiss_index = scraper.store_embeddings(embeddings)
    query_handler = QueryHandler(faiss_index)
    user_query = "What is the significance of RAG in AI?"
    query_embeddings = query_handler.convert_query_to_embeddings(user_query)
    similar_chunks = query_handler.perform_similarity_search(query_embeddings)
    retrieved_chunks = query_handler.retrieve_chunks(similar_chunks)
    response_generator = ResponseGenerator()
    response = response_generator.generate_response(retrieved_chunks)
    print("Response:", response)


if _name_ == "_main_":
    main()
