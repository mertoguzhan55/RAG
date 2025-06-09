# Semantic Chunking RAG

This repository demonstrates how to:

- Convert each line in a `.txt` dataset into sentence embeddings  
- Measure semantic similarity between sentences using cosine similarity  
- Group similar sentences into semantic chunks  
- Store these chunks in a **FAISS** vector database  
- Query the vector store using a user question  
- Use Groq-hosted LLMs (I used llama3-8b-8192) to answer questions based on relevant chunks

---

## Environment Setup
```
conda create -n rag python=3.9
conda activate rag
```

### Requirements

This project runs on **Python 3.9+** with the following dependencies:

```bash
toml==0.10.2
loguru==0.7.3
sentence-transformers==4.1.0
transformers==4.52.3
accelerate==1.7.0
faiss-cpu==1.11.0
langchain==0.3.25
langchain-community==0.3.24
langchain-groq==0.3.2

```

```
pip install -r requirements.txt
```

**EXPORT GROQ API KEY for the language model**
```
export GROQ_API_KEY=groq_api_key
```

### Project Structure
```
.
├── app/
│   ├── config.py
│   ├── logger.py
│   ├── semantic_chunker.py
│   ├── semantic_retriever.py
│   └── app.py
├── dataset/
│   └── only_answer_medquad.txt  #dataset
├── requirements.txt
└── README.md
```



## Run

```
python app.py -env local
```