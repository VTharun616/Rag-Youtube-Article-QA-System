import streamlit as st
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi

# ---------------- GEMINI SETUP ----------------
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

def ask_llm(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",  # safer than 2.5 for now
        contents=prompt
    )
    return response.text

# ---------------- YOUTUBE TRANSCRIPT ----------------
def get_youtube_text(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([item["text"] for item in transcript])
        return text
    except:
        return None

# ---------------- STREAMLIT UI ----------------
st.title("🎥 YouTube RAG QA System")

url = st.text_input("https://youtu.be/T-D1OfcDW1M?si=HWrWE5aY7ecTOBto")

article = None

# ---------------- MAIN LOGIC ----------------
if url:

    try:
        # Extract video ID
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
        else:
            video_id = url.split("v=")[-1].split("&")[0]

        video_link = f"https://www.youtube.com/watch?v={video_id}"

        st.video(video_link)  # ✅ FIX: show video

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

        # ---------------- IMPROVED PROMPT ----------------
        prompt = """
You are a YouTube Video AI assistant.

You will be given transcript text from a YouTube video.

Your job:
1. Understand the video content deeply
2. Generate accurate and useful Q&A
3. Always keep answers grounded ONLY in transcript
4. If possible, refer to parts of the video using context (timestamps or video link if available)
5. Do NOT invent information outside the transcript

Format:
- Clear question
- Short, simple answer
"""

        qa_result = ask_llm(prompt)

        # ---------------- OUTPUT ----------------
        st.subheader("📌 Generated Q&A")
        st.text_area("Output", qa_result, height=300)

        st.markdown("## 🎥 Video Link")
        st.markdown(video_link)

    except Exception as e:
        st.error(f"Error: {e}")
