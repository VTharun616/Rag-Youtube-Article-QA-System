import streamlit as st
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi

# ---------------- GEMINI SETUP ----------------
client = genai.Client(api_key="api_key")  # 🔴 PASTE YOUR KEY HERE

def ask_llm(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ---------------- YOUTUBE TRANSCRIPT ----------------
def get_youtube_text(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item["text"] for item in transcript])
    except:
        return None

# ---------------- STREAMLIT UI ----------------
st.title("🎥 YouTube RAG QA System")

url = st.text_input("https://youtu.be/-46UkLPf9h0?si=FAoHBj6Cz5iGh8iq")

article = None

# ---------------- MAIN LOGIC ----------------
if url:

    try:
        # Extract video ID
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
        else:
            video_id = url.split("v=")[-1].split("&")[0]

        # Get transcript
        with st.spinner("Fetching transcript..."):
            article = get_youtube_text(video_id)

        # Fallback
        if not article:
            st.warning("⚠ No transcript found. Using demo content.")

            article = """
            Generative AI creates text, images, and code.
            LLMs learn patterns from data.
            RAG improves AI using retrieval.
            """

        st.success("System ready ✅")

        # ---------------- Q&A ----------------
        qa_result = ask_llm(f"""
Generate 5 Q&A from this content:

{article}
""")

        st.subheader("📌 Generated Q&A")
        st.text_area("Output", qa_result, height=300)

        st.download_button(
            "⬇ Download Q&A",
            qa_result,
            file_name="qa.txt"
        )

        # ---------------- CHAT ----------------
        st.subheader("💬 Ask Questions")

        user_query = st.text_input("Ask something")

        if user_query:
            response = ask_llm(f"""
Answer ONLY using this content:

{article}

Question:
{user_query}
""")

            st.write("### Answer:")
            st.write(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")
