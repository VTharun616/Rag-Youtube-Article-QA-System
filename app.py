# ---------------- IMPORTS ----------------
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# llm should already be defined (Gemini / OpenAI etc.)

# ---------------- SAMPLE ARTICLE (replace with your actual article/transcript) ----------------
article = article if article else "No article available."

# ---------------- QA SYSTEM PROMPT ----------------
qa_system_message = """
You are an AI assistant.

STRICT RULES:
- Use ONLY the given article
- Do NOT add external knowledge
"""

qa_human_message = """
Generate 5 high-quality questions and answers from the article.

Format:
Q1:
A1:

Q2:
A2:

...

Article:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(qa_system_message),
    HumanMessagePromptTemplate.from_template(qa_human_message)
])

# ---------------- QA CHAIN ----------------
qa_chain = qa_prompt | llm | StrOutputParser()

# ---------------- GENERATE Q&A ----------------
try:
    qa_result = qa_chain.invoke({
        "context": article
    })

    print("\n===== QUESTIONS & ANSWERS =====\n")
    print(qa_result)

    # Save Q&A
    with open("qa.txt", "w", encoding="utf-8") as f:
        f.write(qa_result)

except Exception as e:
    print("Error generating Q&A:", e)


# ---------------- CHAT PROMPT ----------------
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Answer ONLY using the article. Do not add external knowledge."
    ),
    HumanMessagePromptTemplate.from_template(
        "Question: {question}\n\nArticle:\n{context}"
    )
])

chat_chain = chat_prompt | llm | StrOutputParser()

# ---------------- CHAT LOOP ----------------
print("\n🤖 Chatbot ready! Ask questions based on article.")
print("Type 'exit' to stop.\n")

while True:
    user_query = input("Enter your question: ").strip()

    # exit condition
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
        print("Error generating response:", e)
