
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma   # güncel import
from langchain_core.documents import Document
import os
import pandas as pd

# CSV oku
df = pd.read_csv("university_reviews.csv")

# Embedding modeli
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# DB klasörü
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# VectorStore
vector_store = Chroma(
    collection_name="universities",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Eğer ilk defa oluşturuluyorsa dokümanları ekle
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # yorumları birleştir
        reviews = " ".join([
            str(row.get("yorum1", "")),
            str(row.get("yorum2", "")),
            str(row.get("yorum3", ""))
        ])

        document = Document(
            page_content=reviews,
            metadata={
                "üniversite": row["üniversite"],
                "şehir": row["şehir"],
                "sıralama": row["sıralama"]
            }
        )
        ids.append(str(i))
        documents.append(document)

    vector_store.add_documents(documents, ids=ids)

# Retriever oluştur
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
