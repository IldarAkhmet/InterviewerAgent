from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class VectorMemory:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", dim=384):
        self.model = SentenceTransformer(embedding_model_name)
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.entries = []

    def add_entry(self, turn_id: int, question: str, answer: str, review: str, score: int, knowledge_gaps: list, llm_answer):
        text = "Question: " + question + " Answer: " + answer
        embedding = self.model.encode(text)
        self.index.add(np.array([embedding]))
        self.entries.append({
            "turn_id": str(turn_id),

            "question": {
                "role": "system",
                "content": question,
            },

            "answer": {
                "role": "user",
                "content": answer,
            },

            "review": {
                "role": "system",
                "content": review,
                "score": score,
                "knowledge_gaps": knowledge_gaps,
                "llm_answer": llm_answer,
            }
        })

    def search_similar(self, query_text, top_k=3):
        query_vec = self.model.encode(query_text)
        D, I = self.index.search(np.array([query_vec]), top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.entries):
                results.append(self.entries[idx])

        return results

    def get_last_n(self, n=9):
        return self.entries[-n:]
