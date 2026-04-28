import streamlit as st
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi

# ---------------- GEMINI SETUP ----------------
client = genai.Client(api_key="GOOGLE_API_KEY")  # 🔴 replace this

def ask_llm(prompt):
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    return response.text

# ---------------- YOUTUBE TRANSCRIPT FUNCTION ----------------
def get_youtube_text(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([item["text"] for item in transcript])
        return text
    except:
        return None

# ---------------- STREAMLIT UI ----------------
st.title("🎥 Stable RAG YouTube + Article QA System")

url = st.text_input("https://youtu.be/-46UkLPf9h0?si=FAoHBj6Cz5iGh8iq")

article = None

# ---------------- MAIN LOGIC ----------------
if url:

    try:
        # ---------------- Extract Video ID ----------------
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
        else:
            video_id = url.split("v=")[-1].split("&")[0]

        # ---------------- Fetch Transcript ----------------
        with st.spinner("Fetching transcript..."):
            article = get_youtube_text(video_id)

        # ---------------- Fallback ----------------
        if not article:
            st.warning("⚠ No transcript found. Using demo content.")

            article = """
            Generative AI is a branch of artificial intelligence that creates text, images, and code.

            Large Language Models learn from huge datasets.

            Retrieval Augmented Generation improves accuracy using external knowledge.
            """

        st.success("System ready ✅")

        # ---------------- Q&A GENERATION ----------------
        with st.spinner("Generating Q&A..."):
            qa_prompt = f"""
Generate 5 question and answer pairs from this content:

{article}
"""

            qa_result = ask_llm(qa_prompt)

        st.subheader("📌 Generated Q&A")
        st.text_area("Output", qa_result, height=300)

        st.download_button(
            "⬇ Download Q&A",
            qa_result,
            file_name="qa.txt"
        )

        # ---------------- CHAT SECTION ----------------
        st.subheader("💬 Ask Questions")

        user_query = st.text_input("Ask something from the video")

        if user_query:
            chat_prompt = f"""
Answer ONLY using the content below:

CONTENT:
{article}

QUESTION:
{user_query}
"""

            response = ask_llm(chat_prompt)

            st.write("### Answer:")
            st.write(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")
