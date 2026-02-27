# generators/gemini_generator.py
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from generators.base import BaseGenerator, GenerationResult

load_dotenv()

class GeminiGenerator(BaseGenerator):
    def __init__(self, model: str = "gemini-2.0-flash"):
        super().__init__(name=f"Gemini-{model}")
        self.model_name = model
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, query: str, contexts: list[str]) -> GenerationResult:
        context_block = "\n\n---\n\n".join(contexts)

        prompt = f"""You are a helpful AI assistant. Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.

CONTEXT:
{context_block}

QUESTION:
{query}

ANSWER:"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024
            )
        )

        answer = response.text
        tokens = len(prompt.split()) + len(answer.split())

        return GenerationResult(
            answer=answer,
            latency_ms=0,
            tokens_used=tokens,
            metadata={"model": self.model_name}
        )