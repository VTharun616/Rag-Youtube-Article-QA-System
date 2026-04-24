# ---------------- IMPORTS ----------------
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import RequestBlocked, TranscriptsDisabled

# llm must already be defined (Gemini / OpenAI etc.)

# ---------------- GET YOUTUBE TRANSCRIPT ----------------
def get_youtube_text(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t["text"] for t in transcript])
        return text

    except (RequestBlocked, TranscriptsDisabled):
        return ""

    except Exception:
        return ""


# ---------------- INPUT ----------------
youtube_url = input("https://youtu.be/-46UkLPf9h0?si=FAoHBj6Cz5iGh8iq")

# extract video id (simple version)
video_id = youtube_url.split("v=")[-1].split("&")[0]

article = get_youtube_text(video_id)

# fallback safety
if not article:
    article = "No transcript available for this video."


# ---------------- QA PROMPT ----------------
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

...

Content:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(qa_system_message),
    HumanMessagePromptTemplate.from_template(qa_human_message)
])

qa_chain = qa_prompt | llm | StrOutputParser()


# ---------------- GENERATE Q&A ----------------
try:
    qa_result = qa_chain.invoke({
        "context": article
    })

    print("\n===== QUESTIONS & ANSWERS =====\n")
    print(qa_result)

    with open("qa.txt", "w", encoding="utf-8") as f:
        f.write(qa_result)

except Exception as e:
    print("QA generation error:", e)


# ---------------- CHAT PROMPT ----------------
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Answer ONLY using the provided content. Do not use external knowledge."
    ),
    HumanMessagePromptTemplate.from_template(
        "Question: {question}\n\nContent:\n{context}"
    )
])

chat_chain = chat_prompt | llm | StrOutputParser()


# ---------------- CHAT LOOP ----------------
print("\n🤖 Chatbot ready! Ask questions based on YouTube video.")
print("Type 'exit' to stop.\n")

while True:
    user_query = input("Ask: ").strip()

    if user_query.lower() in ["exit", "quit", "bye"]:
        print("Goodbye 👋")
        break

    try:
        response = chat_chain.invoke({
            "question": user_query,
            "context": article
        })

        print("\nAnswer:\n", response, "\n")

    except Exception as e:
        print("Error:", e)
