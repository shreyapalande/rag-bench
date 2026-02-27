# generators/groq_generator.py
import os
from groq import Groq
from dotenv import load_dotenv
from generators.base import BaseGenerator, GenerationResult

load_dotenv()

class GroqGenerator(BaseGenerator):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        super().__init__(name=f"Groq-{model}")
        self.model = model
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate(self, query: str, contexts: list[str]) -> GenerationResult:
        # Combine retrieved chunks into context block
        context_block = "\n\n---\n\n".join(contexts)

        prompt = f"""You are a helpful AI assistant. Answer the question using ONLY the provided context.
        If the context doesn't contain enough information, say so clearly.

        CONTEXT:
        {context_block}

        QUESTION:
        {query}

        ANSWER:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512
        )

        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens

        return GenerationResult(
            answer=answer,
            latency_ms=0,
            tokens_used=tokens,
            metadata={"model": self.model}
        )
