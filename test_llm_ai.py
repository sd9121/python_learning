# chatbot_runner.py

from docsearch_file import create_docsearch  # ✅ Import your docsearch setup
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import time
import os

# === STEP 1: Load Pinecone docsearch ===
docsearch = create_docsearch()
retriever = docsearch.as_retriever()

# === STEP 2: Initialize Chat LLM (GPT-4o-mini) ===
llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model_name='gpt-4o-mini',
    temperature=0.0
)

# === STEP 3: Set up the Retrieval-Augmented Generation Chain ===
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# === STEP 4: Define the user queries ===
query1 = "What are the first 3 steps for getting started with the WonderVector5000?"
query2 = "The Neural Fandango Synchronizer is giving me a headache. What do I do?"

# === Step 5: Run query WITHOUT Pinecone context (just the base LLM) ===
answer1_without_knowledge = llm.invoke(query1)
print("Query 1:", query1)
print("\nAnswer WITHOUT Pinecone context:\n", answer1_without_knowledge.content)
print("\n")
time.sleep(2)

answer2_without_knowledge = llm.invoke(query2)
print("Query 2:", query2)
print("\nAnswer WITHOUT Pinecone context:\n", answer2_without_knowledge.content)
print("\n")
time.sleep(2)

# === Step 6: Run query WITH Pinecone context using RAG ===
answer1_with_knowledge = retrieval_chain.invoke({"input": query1})
print("Answer WITH Pinecone knowledge:\n", answer1_with_knowledge['answer'])
print("\nContext Used:\n", answer1_with_knowledge['context'])
print("\n")
time.sleep(2)

answer2_with_knowledge = retrieval_chain.invoke({"input": query2})
print("Answer WITH Pinecone knowledge:\n", answer2_with_knowledge['answer'])
print("\nContext Used:\n", answer2_with_knowledge['context'])
print("\n")
