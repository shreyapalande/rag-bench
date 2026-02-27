# evaluation/ragas_eval.py
import json
import pandas as pd
from dataclasses import dataclass
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str

@dataclass 
class EvalResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    
    def average(self) -> float:
        return (
            self.faithfulness +
            self.answer_relevancy +
            self.context_precision +
            self.context_recall
        ) / 4

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "average": round(self.average(), 3)
        }

class RAGASEvaluator:
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

    def evaluate_samples(self, samples: list[EvalSample]) -> EvalResult:
        """Run RAGAS evaluation on a list of samples."""
        
        # Convert to HuggingFace Dataset format RAGAS expects
        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth for s in samples]
        }
        dataset = Dataset.from_dict(data)

        # Run evaluation
        results = evaluate(dataset, metrics=self.metrics)
        df = results.to_pandas()

        # Average across all samples
        return EvalResult(
            faithfulness=df["faithfulness"].mean(),
            answer_relevancy=df["answer_relevancy"].mean(),
            context_precision=df["context_precision"].mean(),
            context_recall=df["context_recall"].mean()
        )

    def load_ground_truth(self, path: str) -> dict:
        """Load ground truth file as question->answer dict."""
        with open(path, "r") as f:
            data = json.load(f)
        return {item["question"]: item["ground_truth"] for item in data}