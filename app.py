import streamlit as st
from google import genai

# ---------------- GEMINI SETUP ----------------
client = genai.Client(api_key="GOOGLE_API_KEY")  # replace this

def ask_llm(prompt):
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    return response.text

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

        # ---------------- Get Transcript ----------------
        with st.spinner("Fetching transcript..."):
            article = get_youtube_text(video_id)   # your function

        # ---------------- FALLBACK ----------------
        if not article:
            st.warning("⚠ YouTube blocked or no transcript found. Using demo content.")

            article = """
            Generative AI is a branch of AI that creates text, images, and code.

            Large Language Models (LLMs) learn patterns from large datasets.

            Retrieval Augmented Generation (RAG) improves accuracy using external knowledge retrieval.
            """

        st.success("System ready ✅")

        # ---------------- Q&A GENERATION ----------------
        with st.spinner("Generating Q&A..."):
            qa_result = ask_llm(f"""
Generate 5 question and answer pairs from the following content:

{article}
""")

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
            response = ask_llm(f"""
Answer ONLY using the given content.

CONTENT:
{article}

QUESTION:
{user_query}
""")

            st.write("### Answer:")
            st.write(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")
