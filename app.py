from app.config import Configs
from app.logger import Logger
from app.semantic_chunker import SemanticChunker
from app.semantic_retriever import SemanticRetriever
from app.llm import LLM

def main(args, configs):

    logger = Logger(**configs["logger"])
    logger.debug("############ [NAME OF PROJECT] CONFIGURATIONS ############")
    logger.debug(configs)


    # Converts each sentence in given dataset to an embedding and compares it with others using cosine similarity.
    # Sentences with similarity above the threshold are grouped into the same chunk.
    semantic_chunker = SemanticChunker(**configs["semantic_chunker"])
    semantic_chunker.run()
    
    retriever = SemanticRetriever(**configs["semantic_retriever"])
    query = "What are the different types of diabetes?"
    retrieved_chunks = retriever.retrieve(query)

    prompt = (
            "The following texts contain information related to health. Based on this information, please provide a detailed answer to the question below.\n\n"
            + "\n\n".join(retrieved_chunks) +
            f"\n\nQuestion: {query}\nPlease answer in detail:"
        )

    llm = LLM(**configs["llm"])
    response = llm.ask(prompt)
    print(response)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str)
    args = parser.parse_args()

    configs = Configs().load(config_name=args.environment)
    main(args, configs)
