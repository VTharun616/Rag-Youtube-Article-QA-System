# STREAMLIT LINK
https://multisourceragproject-7nyt4habgkjhmsusvvqppv.streamlit.app/

## 1. Overview
This project is a Retrieval-Augmented Generation (RAG) system that allows users to ask questions from PDF documents and web pages. 
It retrieves relevant context using vector similarity search and generates accurate answers using Google Gemini LLM. 
The system also includes a routing mechanism to decide whether the query should be answered from PDF or web data.

## 2. Features
* Load and process PDF documents
* Scrape and process web pages
* Text chunking using RecursiveCharacterTextSplitter
* Embeddings using Sentence Transformers
* Vector storage using FAISS
*Query routing (PDF or Web source selection)
*Context-based question answering using Gemini
*Hallucination control (answers strictly from context)
*Interactive CLI-based Q&A system

## 3. How It Works
User Query → Query Router → Embedding Generation → FAISS Vector Search → Relevant Context Retrieval → Prompt Construction → Gemini LLM → Final Answer

## 4. Tech Stack
* Python
* LangChain
*Google Gemini (gemini-2.5-flash)
* HuggingFace Sentence Transformers
* FAISS Vector Database
* PyPDF
*WebBaseLoader
*Google Colab

