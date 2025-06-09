from dataclasses import dataclass
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


@dataclass
class SemanticChunker:
    dataset_path: str
    embedding_model_name: str
    threshold: float
    vector_db_path: str

    def splitter_by_line(self) -> List[str]:
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file.readlines()]

    def create_embeddings(self, sentences: List[str]):
        model = SentenceTransformer(self.embedding_model_name)
        return model.encode(sentences, show_progress_bar=True)

    def group_sentences_by_similarity(self, sentences: List[str], embeddings: List[List[float]]) -> List[str]:
        similarity_matrix = cosine_similarity(embeddings)
        visited = set()
        chunks = []

        for i in range(len(sentences)):
            if i in visited:
                continue

            current_chunk = [sentences[i]]
            visited.add(i)

            for j in range(len(sentences)):
                if j != i and j not in visited and similarity_matrix[i][j] >= self.threshold:
                    current_chunk.append(sentences[j])
                    visited.add(j)

            chunks.append("\n".join(current_chunk))

        return chunks

    def save_chunks_to_vector_db(self, chunks: List[str]):
        embedding_func = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        db = FAISS.from_texts(chunks, embedding=embedding_func)
        db.save_local(self.vector_db_path)

    def run(self):
        sentences = self.splitter_by_line()

        embeddings = self.create_embeddings(sentences)
        chunks = self.group_sentences_by_similarity(sentences, embeddings)

        self.save_chunks_to_vector_db(chunks)



if __name__ == "__main__":
    pass
