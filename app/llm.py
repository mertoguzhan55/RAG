from langchain_groq import ChatGroq
from dataclasses import dataclass
import os

@dataclass
class LLM:
    model_name: str
    max_tokens: int

    def ask(self, prompt):

        llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name=self.model_name,
                max_tokens=self.max_tokens
            )
        response = llm.invoke(prompt)
        return str(response.content)