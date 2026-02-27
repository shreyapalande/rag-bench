# evaluation/llm_judge.py
import json
import os
from dataclasses import dataclass, field
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

@dataclass
class JudgeScore:
    faithfulness: float        # Is answer supported by context?
    answer_relevancy: float    # Does answer address the question?
    context_relevancy: float   # Are retrieved chunks relevant?
    completeness: float        # Does answer cover ground truth?
    reasoning: dict = field(default_factory=dict)

    def average(self) -> float:
        return (
            self.faithfulness +
            self.answer_relevancy +
            self.context_relevancy +
            self.completeness
        ) / 4

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "context_relevancy": round(self.context_relevancy, 3),
            "completeness": round(self.completeness, 3),
            "average": round(self.average(), 3)
        }

class LLMJudge:
    """
    LLM-as-judge evaluator using Groq LLaMA.
    Evaluates RAG outputs across 4 dimensions without needing OpenAI.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def _call_judge(self, prompt: str) -> dict:
        """Call LLM and parse JSON response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512
        )
        raw = response.choices[0].message.content.strip()

        # Extract JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in response: {raw}")
        return json.loads(raw[start:end])

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str
    ) -> JudgeScore:
        """Evaluate a single RAG output across all 4 dimensions."""

        context_block = "\n---\n".join(contexts)

        prompt = f"""You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Evaluate the following RAG output across 4 dimensions. Return ONLY a JSON object, no other text.

QUESTION: {question}

RETRIEVED CONTEXT:
{context_block}

GENERATED ANSWER: {answer}

GROUND TRUTH ANSWER: {ground_truth}

Evaluate each dimension from 0.0 to 1.0:

1. faithfulness: Are all claims in the generated answer supported by the retrieved context? 
   (1.0 = fully supported, 0.0 = contradicts or ignores context)

2. answer_relevancy: Does the generated answer actually address the question asked?
   (1.0 = directly answers, 0.0 = completely off-topic)

3. context_relevancy: Are the retrieved context chunks relevant to the question?
   (1.0 = highly relevant, 0.0 = completely irrelevant)

4. completeness: Does the generated answer cover the key points in the ground truth?
   (1.0 = covers all key points, 0.0 = misses everything important)

Return this exact JSON format:
{{
    "faithfulness": 0.0,
    "answer_relevancy": 0.0,
    "context_relevancy": 0.0,
    "completeness": 0.0,
    "reasoning": {{
        "faithfulness": "brief reason",
        "answer_relevancy": "brief reason",
        "context_relevancy": "brief reason",
        "completeness": "brief reason"
    }}
}}"""

        result = self._call_judge(prompt)

        return JudgeScore(
            faithfulness=float(result["faithfulness"]),
            answer_relevancy=float(result["answer_relevancy"]),
            context_relevancy=float(result["context_relevancy"]),
            completeness=float(result["completeness"]),
            reasoning=result.get("reasoning", {})
        )

    def evaluate_batch(
        self,
        samples: list[dict]
    ) -> tuple[JudgeScore, list[JudgeScore]]:
        """
        Evaluate a batch of samples.
        Returns (averaged_score, individual_scores).
        """
        individual_scores = []

        for i, sample in enumerate(samples):
            print(f"  Judging sample {i+1}/{len(samples)}...")
            score = self.evaluate(
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample["contexts"],
                ground_truth=sample["ground_truth"]
            )
            individual_scores.append(score)

        # Average across all samples
        avg = JudgeScore(
            faithfulness=sum(s.faithfulness for s in individual_scores) / len(individual_scores),
            answer_relevancy=sum(s.answer_relevancy for s in individual_scores) / len(individual_scores),
            context_relevancy=sum(s.context_relevancy for s in individual_scores) / len(individual_scores),
            completeness=sum(s.completeness for s in individual_scores) / len(individual_scores)
        )

        return avg, individual_scores