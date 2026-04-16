from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def create_vector_db():
    loader = TextLoader("data/data.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="db"
    )

    db.persist()
    return db


def load_vector_db():
    embeddings = HuggingFaceEmbeddings()
    return Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )
