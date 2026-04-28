import streamlit as st
import os

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ---------------- UI ----------------
st.title("🎥 Stable RAG YouTube + Article QA System")

# ---------------- SAFE YOUTUBE FUNCTION ----------------
def get_youtube_text(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except:
        return None

# ---------------- INPUT ----------------
url = st.text_input("https://youtu.be/-46UkLPf9h0?si=FAoHBj6Cz5iGh8iq")

article = None

# ---------------- MAIN LOGIC ----------------
if url:
    try:
        video_id = url.split("v=")[-1].split("&")[0]

        with st.spinner("Fetching transcript..."):
            article = get_youtube_text(video_id)

        # ---------------- FALLBACK SYSTEM (VERY IMPORTANT) ----------------
        if not article:
            st.warning("⚠ YouTube blocked or no transcript found. Using demo content.")

            article = """
            Generative AI is a branch of artificial intelligence that creates new content such as text, images, and code.

            Large Language Models (LLMs) like GPT are trained on massive datasets to understand and generate human-like text.

            Retrieval Augmented Generation (RAG) improves AI responses by retrieving relevant documents before generating answers.
            """

        st.success("System ready ✅")

        # ---------------- QA PROMPT ----------------
        qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an AI assistant. Use ONLY the given content."
            ),
            HumanMessagePromptTemplate.from_template(
                "Generate 5 Q&A from this content:\n\n{context}"
            )
        ])

        qa_chain = qa_prompt | llm | StrOutputParser()

        with st.spinner("Generating Q&A..."):
            qa_result = qa_chain.invoke({"context": article})

        st.subheader("📌 Generated Q&A")
        st.text_area("Output", qa_result, height=300)

        st.download_button(
            "⬇ Download Q&A",
            qa_result,
            file_name="qa.txt"
        )

        # ---------------- CHAT SECTION ----------------
        st.subheader("💬 Ask Questions")

        user_query = st.text_input("Ask something from the content")

        if user_query:
            chat_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "Answer ONLY using the given content."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Question: {question}\n\nContent:\n{context}"
                )
            ])

            chat_chain = chat_prompt | llm | StrOutputParser()

            response = chat_chain.invoke({
                "question": user_query,
                "context": article
            })

            st.write("### Answer:")
            st.write(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")
