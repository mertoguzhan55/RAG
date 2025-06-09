from dataclasses import dataclass
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

@dataclass
class SemanticRetriever:
    vector_db_path: str
    embedding_model_name: str
    top_k: int

    def retrieve(self, query: str, top_k: int = 3):
        embedding_func = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        db = FAISS.load_local(self.vector_db_path, embeddings = embedding_func, allow_dangerous_deserialization=True)
        results = db.similarity_search_with_score(query, k=self.top_k)

        return [doc.page_content for doc, _ in results]