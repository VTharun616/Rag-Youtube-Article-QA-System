import os
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi

# -----------------------------
# API KEY
# -----------------------------
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Missing GOOGLE_API_KEY in Secrets")
    st.stop()

# -----------------------------
# LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# -----------------------------
# EMBEDDINGS
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# GET TRANSCRIPT
# -----------------------------
def get_text(video_url):
    video_id = video_url.split("v=")[-1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([t["text"] for t in transcript])

# -----------------------------
# UI
# -----------------------------
st.title("🎥 YouTube RAG Chatbot")

url = st.text_input("Paste YouTube Link")

if url:
    text = get_text(url)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([text])

    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(k=4)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask something about video")

    if query:
        st.session_state.chat.append({"role": "user", "content": query})

        docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs])

        prompt = f"""
        Answer using only this video content:

        {context}

        Question: {query}
        """

        response = llm.invoke(prompt)

        st.session_state.chat.append(
            {"role": "assistant", "content": response.content}
        )

        with st.chat_message("assistant"):
            st.markdown(response.content)
