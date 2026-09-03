# src/test.py

import json
import re
import requests


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

# Ollama judge model
JUDGE_MODEL = "qwen2.5:1.5b"

# Choose which generation result to evaluate
# 0 base model, 1 lora, 2 qlora
MODE = 2
MODEL_TYPE = "base" if MODE == 0 else "lora" if MODE == 1 else "qlora"

INPUT_FILE = f"results/{MODEL_TYPE}_generation.json"
OUTPUT_FILE = f"results/{MODEL_TYPE}_evaluation.json"


# ============================================================
# Load generation results
# ============================================================

def load_results():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# Ollama LLM Judge
# ============================================================

def llm_judge(
    question,
    reference,
    prediction
):
    prompt = f"""You are an evaluator for a question-answering system.

Determine whether the generated answer correctly answers the question.

Question:
{question}

Reference Answer:
{reference}

Generated Answer:
{prediction}

Rules:

- Return "Correct" if the generated answer is factually correct and
  adequately answers the question.
- Minor omissions, differences in wording, or reasonable paraphrasing
  are acceptable.
- The generated answer does not need to match the reference answer
  word-for-word.
- Return "Wrong" if the answer contains a factual error, is irrelevant,
  contradicts the reference, or is substantially incomplete.
- If the question asks for multiple important steps or components,
  missing several major steps should be considered Wrong.

Return ONLY one word:
Correct
Wrong
"""

    payload = {
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        judgement = result["response"].strip()

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        normalized = judgement.lower().strip()

        if normalized.startswith("correct"):
            return "Correct"

        if normalized.startswith("wrong"):
            return "Wrong"

        # ----------------------------------------------------
        # Fallback regex
        # ----------------------------------------------------

        match = re.search(
            r"\b(correct|wrong)\b",
            normalized
        )

        if match:

            if match.group(1) == "correct":
                return "Correct"

            if match.group(1) == "wrong":
                return "Wrong"

        return "Judge Error"

    except Exception as e:

        print(
            f"LLM Judge Error: {e}"
        )

        return "Judge Error"


# ============================================================
# Evaluation
# ============================================================

def evaluate():

    data = load_results()

    results = data["results"]

    correct = 0
    wrong = 0
    errors = 0

    evaluated_results = []

    for i, item in enumerate(results, 1):

        print(
            f"\n[{i}/{len(results)}] Judging..."
        )

        question = item["question"]
        reference = item["reference"]
        prediction = item["prediction"]

        judgement = llm_judge(
            question,
            reference,
            prediction
        )

        # ----------------------------------------------------
        # Count
        # ----------------------------------------------------

        if judgement == "Correct":
            correct += 1

        elif judgement == "Wrong":
            wrong += 1

        else:
            errors += 1

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        evaluated_results.append({

            "id": item["id"],

            "question": question,

            "reference": reference,

            "prediction": prediction,

            "llm_judge": judgement

        })

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print("-" * 80)

        print(
            f"Question: {question}"
        )

        print(
            f"Prediction: {prediction}"
        )

        print(
            f"Reference: {reference}"
        )

        print(
            f"LLM Judge: {judgement}"
        )

        print(
            f"Accuracy: "
            f"{correct}/{i} = "
            f"{correct / i:.2%}"
        )

        print("-" * 80, flush=True)

    # ========================================================
    # Final result
    # ========================================================

    accuracy = (
        correct / len(results)
        if len(results) > 0
        else 0
    )

    output = {

        "experiment": data.get(
            "experiment",
            MODEL_TYPE
        ),

        "model": data.get(
            "model",
            "unknown"
        ),

        "judge_model": JUDGE_MODEL,

        "num_samples": len(results),

        "correct": correct,

        "wrong": wrong,

        "judge_errors": errors,

        "accuracy": accuracy,

        "results": evaluated_results

    }

    # ========================================================
    # Save
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 80)

    print(
        f"Model: {MODEL_TYPE}"
    )

    print(
        f"Judge: {JUDGE_MODEL}"
    )

    print(
        f"Correct: {correct}/{len(results)}"
    )

    print(
        f"Wrong: {wrong}/{len(results)}"
    )

    print(
        f"Judge Errors: {errors}"
    )

    print(
        f"Accuracy: {accuracy:.2%}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("=" * 80)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    evaluate()
