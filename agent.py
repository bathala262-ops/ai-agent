from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from rag import load_vector_db

# Load DB
db = load_vector_db()
retriever = db.as_retriever()

# Load LLM
llm = Ollama(model="mistral")

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Tool
def search_docs(query):
    return qa_chain.run(query)

tools = [
    Tool(
        name="Knowledge Base",
        func=search_docs,
        description="Answer questions from stored documents"
    )
]

# Agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=False
)

def ask_agent(query):
    return agent.run(query)
