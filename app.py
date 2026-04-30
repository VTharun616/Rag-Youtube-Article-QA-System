import os
import streamlit as st

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# API KEY
# -----------------------------
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY missing in Secrets")
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
# ROBUST TRANSCRIPT FUNCTION (FIXED)
# -----------------------------
def get_transcript(video_url):
    try:
        # extract video id
        if "v=" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0]
        else:
            video_id = video_url.split("/")[-1]

        transcript = None

        # STEP 1: normal fetch
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)

        except Exception:
            # STEP 2: fallback to ALL transcripts (IMPORTANT FIX)
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            for t in transcript_list:
                transcript = t.fetch()
                break

        if not transcript:
            return None, "❌ No transcript available"

        text = " ".join([t["text"] for t in transcript])
        return text, None

    except TranscriptsDisabled:
        return None, "❌ Transcripts are disabled for this video"

    except NoTranscriptFound:
        return None, "❌ No transcript found"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# -----------------------------
# BUILD VECTOR DB
# -----------------------------
def build_db(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([text])
    db = FAISS.from_documents(chunks, embeddings)
    return db

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎥 YouTube QA RAG System (Fixed Version)")

st.info("👉 Use educational videos with captions for best results")

url = st.text_input("Paste YouTube Link")

if url:

    with st.spinner("Processing video... ⏳"):

        text, error = get_transcript(url)

        if error:
            st.error(error)
            st.stop()

        db = build_db(text)
        retriever = db.as_retriever(search_kwargs={"k": 4})

        st.success("✅ Video loaded successfully!")

        if "chat" not in st.session_state:
            st.session_state.chat = []

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("Ask something about the video")

        if query:
            st.session_state.chat.append({"role": "user", "content": query})

            docs = retriever.invoke(query)
            context = "\n".join([d.page_content for d in docs])

            prompt = f"""
            Answer ONLY using the context below:

            {context}

            Question: {query}
            """

            response = llm.invoke(prompt)

            st.session_state.chat.append(
                {"role": "assistant", "content": response.content}
            )

            with st.chat_message("assistant"):
                st.markdown(response.content)
