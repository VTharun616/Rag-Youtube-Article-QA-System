import os
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- LLM (FIXED - REQUIRED) ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ---------------- STREAMLIT UI ----------------
st.title("🎥 RAG YouTube + Article QA System")

# ---------------- SAFE YOUTUBE LOADER ----------------
def get_youtube_text(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except Exception:
        return ""

# ---------------- PROMPTS ----------------
qa_system_message = """
You are an AI assistant.

STRICT RULES:
- Use ONLY the given content
- Do NOT add external knowledge
"""

qa_human_message = """
Generate 5 high-quality questions and answers.

Format:
Q1:
A1:

Q2:
A2:

Q3:
A3:

Q4:
A4:

Q5:
A5:

Content:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(qa_system_message),
    HumanMessagePromptTemplate.from_template(qa_human_message)
])

qa_chain = qa_prompt | llm | StrOutputParser()

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Answer ONLY using the given content. Do not use external knowledge."
    ),
    HumanMessagePromptTemplate.from_template(
        "Question: {question}\n\nContent:\n{context}"
    )
])

chat_chain = chat_prompt | llm | StrOutputParser()

# ---------------- INPUT ----------------
url = st.text_input("https://youtu.be/NRmAXDWJVnU?si=YoiiX2Sn3aksFlxJ")

# ---------------- MAIN FLOW ----------------
if url:
    try:
        video_id = url.split("v=")[-1].split("&")[0]

        with st.spinner("Fetching transcript..."):
            article = get_youtube_text(video_id)

        if not article:
            st.error("❌ Transcript not available for this video.")
        else:
            st.success("✅ Transcript loaded successfully")

            # ---------------- QA GENERATION ----------------
            with st.spinner("Generating Q&A..."):
                qa_result = qa_chain.invoke({"context": article})

            st.subheader("📌 Generated Q&A")
            st.text_area("Q&A Output", qa_result, height=300)

            st.download_button(
                label="⬇ Download Q&A",
                data=qa_result,
                file_name="qa.txt",
                mime="text/plain"
            )

            # ---------------- CHAT SECTION ----------------
            st.subheader("💬 Ask Questions")

            user_query = st.text_input("Ask something from the video")

            if user_query:
                with st.spinner("Thinking..."):
                    response = chat_chain.invoke({
                        "question": user_query,
                        "context": article
                    })

                st.write("### Answer:")
                st.write(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")
