from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# LLM
llm = OllamaLLM(model="llama3")

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load DB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
""")

# RAG pipeline
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

query = input("Ask a question: ")

result = rag_chain.invoke(query)

print("\nAnswer:")
print(result)