import os
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# -----------------------------
# API KEY
# -----------------------------
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found in Secrets")
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
# GET YOUTUBE TRANSCRIPT (FIXED)
# -----------------------------
def get_text(video_url):
    try:
        if "youtube.com" not in video_url and "youtu.be" not in video_url:
            return "Invalid YouTube URL"

        # extract video id safely
        if "v=" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0]
        else:
            video_id = video_url.split("/")[-1]

        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        text = " ".join([t["text"] for t in transcript])
        return text

    except TranscriptsDisabled:
        return "Transcript is disabled for this video."

    except NoTranscriptFound:
        return "No transcript found for this video."

    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎥 YouTube QA RAG System")

url = st.text_input("Paste YouTube Link")

if url:
    with st.spinner("Processing video..."):
        text = get_text(url)

        if "Error" in text or "Transcript" in text:
            st.error(text)
            st.stop()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.create_documents([text])

        db = FAISS.from_documents(chunks, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 4})

        st.success("Video loaded successfully! Ask questions below 👇")

        if "chat" not in st.session_state:
            st.session_state.chat = []

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("Ask anything about the video")

        if query:
            st.session_state.chat.append({"role": "user", "content": query})

            docs = retriever.invoke(query)
            context = "\n".join([d.page_content for d in docs])

            prompt = f"""
            Answer only using the video context below.

            Context:
            {context}

            Question: {query}
            """

            response = llm.invoke(prompt)

            st.session_state.chat.append(
                {"role": "assistant", "content": response.content}
            )

            with st.chat_message("assistant"):
                st.markdown(response.content)
