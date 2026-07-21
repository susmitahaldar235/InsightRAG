import os
import sys
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from modules.rag_service import RAGService

service = RAGService()

questions_path = (
    ROOT_DIR
    / "evaluation"
    / "dataset"
    / "baseline_questions.json"
)

output_path = ROOT_DIR / "evaluation" / "results"
output_path.mkdir(parents=True, exist_ok=True)

BASELINE_FILE = output_path / "baseline_answers.json"
CORRECTIVE_FILE = output_path / "corrective_answers.json"

# ---------------------------------------------------
# Load Questions
# ---------------------------------------------------

with open(questions_path, "r", encoding="utf-8") as f:
    questions = json.load(f)

# ---------------------------------------------------
# Load Existing Results
# ---------------------------------------------------

if BASELINE_FILE.exists():
    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        baseline_results = json.load(f)
else:
    baseline_results = []

if CORRECTIVE_FILE.exists():
    with open(CORRECTIVE_FILE, "r", encoding="utf-8") as f:
        corrective_results = json.load(f)
else:
    corrective_results = []

completed_baseline = {
    item["question"]
    for item in baseline_results
}

completed_corrective = {
    item["question"]
    for item in corrective_results
}

# ---------------------------------------------------
# Evaluate
# ---------------------------------------------------

for item in questions:

    question = item["question"]

    print("\n--------------------------------------")
    print(f"Asking: {question}")

  

    # ===================================================
    # Corrective RAG
    # ===================================================

    if question not in completed_corrective:

        try:
            print("Running Corrective...")

            start = time.time()

            corrective_answer, corrective_docs = service.answer(question)

            corrective_latency = time.time() - start

            corrective_results.append(
                {
                    "id": item["id"],
                    "question": question,
                    "generated_answer": corrective_answer,
                    "retrieved_contexts": [
                        doc["document"] for doc in corrective_docs
                    ],
                    "sources": [
                        doc["metadata"]["source"] for doc in corrective_docs
                    ],
                    "latency": corrective_latency,
                }
            )

            with open(CORRECTIVE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    corrective_results,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

            print(" Corrective saved.")

            time.sleep(15)

        except Exception as e:
            print("\n Corrective stopped.")
            print(e)
            break

    else:
        print(" Corrective already completed.")

print("\n🎉 Evaluation finished (or paused due to quota).")
print("You can safely rerun this script anytime.")