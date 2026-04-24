# ---------------- IMPORTS ----------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ---------------- LLM (ADD YOUR API KEY HERE) ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyAs0N2ZZuHkdHskrghyjO2FWy7GRUeqmXM"
)


# ---------------- LOAD YOUTUBE ----------------
loader = YoutubeLoader.from_youtube_url("https://youtu.be/-46UkLPf9h0?si=FAoHBj6Cz5iGh8iq")
docs = loader.load()


# ---------------- SPLIT ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)


# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# ---------------- VECTOR STORE ----------------
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)


# ---------------- RETRIEVER ----------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ---------------- ARTICLE PROMPT ----------------
system_message = """
You are a technical writer.

STRICT RULES:
- Use ONLY the given context
- Do NOT add external knowledge
- Write clean structured article
"""

human_message = """
Convert the content into a structured article.

Question:
{question}

Context:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    HumanMessagePromptTemplate.from_template(human_message)
])


# ---------------- ARTICLE CHAIN ----------------
chain = (
    {
        "context": retriever | (lambda docs: "\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)


# ---------------- GENERATE ARTICLE ----------------
question = "Explain DoRA and how it improves over LoRA with key differences"
article = chain.invoke(question)

print("\n===== ARTICLE =====\n")
print(article)


# ---------------- SAVE ARTICLE ----------------
with open("article.txt", "w", encoding="utf-8") as f:
    f.write(article)


# ---------------- QA PROMPT ----------------
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
qa_result = qa_chain.invoke({
    "context": article
})

print("\n===== QUESTIONS & ANSWERS =====\n")
print(qa_result)


# ---------------- SAVE Q&A ----------------
with open("qa.txt", "w", encoding="utf-8") as f:
    f.write(qa_result)


# ---------------- CHAT LOOP ----------------
print("\n🤖 Chatbot ready! Ask questions based on article.")
print("Type 'exit' to stop.\n")

while True:
    user_query = input("Enter your question: ").strip()

    if user_query.lower() in ["exit", "quit", "bye"]:
        print("Goodbye 👋")
        break

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Answer ONLY using the article. Do not add external info."
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: {question}\n\nArticle:\n{context}"
        )
    ])

    chat_chain = chat_prompt | llm | StrOutputParser()

    response = chat_chain.invoke({
        "question": user_query,
        "context": article
    })

    print("\nAnswer:\n", response, "\n")