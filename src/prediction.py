# src/evaluate.py

import json

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from transformers import BitsAndBytesConfig

from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# 0 base model, 1 lora, 2 qlora
MODE = 2


if MODE == 0:
    NEED_QUANT = False
    NEED_SFT = False

elif MODE == 1:
    NEED_QUANT = False
    NEED_SFT = True

elif MODE == 2:
    NEED_QUANT = True
    NEED_SFT = True


def load_test():

    with open(
        "data/test.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def load_model():
    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    # --------------------------------------------------
    # Load base model
    # --------------------------------------------------
    quantization_config = None
    if NEED_QUANT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # --------------------------------------------------
    # Load LoRA / QLoRA adapter
    # --------------------------------------------------

    if NEED_SFT:
        adapter_dir = (
            "output/qlora" if NEED_QUANT
            else "output/lora"
        )

        model = PeftModel.from_pretrained(
            base_model,
            adapter_dir,
        )

    else:

        model = base_model

    model.eval()

    return tokenizer, model


def generate(model, tokenizer, question):

    # --------------------------------------------------
    # Build the same prompt format used during training
    # --------------------------------------------------

    prompt = f"Question: {question}\nAnswer:"

    # --------------------------------------------------
    # Tokenize
    # --------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    print(
        "PROMPT =",
        repr(prompt)
    )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=128,

            # Deterministic generation
            do_sample=False,
        )

    # --------------------------------------------------
    # Remove original prompt
    # --------------------------------------------------

    generated = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    # --------------------------------------------------
    # Decode
    # --------------------------------------------------

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return answer


def evaluate():

    # --------------------------------------------------
    # 1. Load test data
    # --------------------------------------------------

    data = load_test()

    # --------------------------------------------------
    # 2. Load model
    # --------------------------------------------------

    tokenizer, model = load_model()

    # --------------------------------------------------
    # 3. Generate answers
    # --------------------------------------------------

    results = []

    for i, item in enumerate(data, 1):

        print(
            f"\n[{i}/{len(data)}] Generating..."
        )

        question = item["question"]
        reference = item["reference_answer"]

        answer = generate(
            model,
            tokenizer,
            question
        )

        # --------------------------------------------------
        # Save raw generation result
        # --------------------------------------------------

        results.append({

            "id": int(item["id"]),

            "question": str(question),

            "reference": str(reference),

            "prediction": str(answer),

        })

        # --------------------------------------------------
        # Show result
        # --------------------------------------------------

        print("-" * 80)

        print(f"Question: {question}")

        print(f"Prediction: {answer}")

        print(f"Reference: {reference}")

        print("-" * 80, flush=True)

    # --------------------------------------------------
    # 4. Determine model type
    # --------------------------------------------------
    if MODE == 0:
        model_type = "base"
    elif MODE == 1:
        model_type = "lora"
    else:
        model_type = "qlora"

    # --------------------------------------------------
    # 5. Save JSON
    # --------------------------------------------------
    result = {
        "experiment": model_type,
        "model": MODEL_NAME,
        "num_samples": len(results),
        "results": results
    }

    output_file = (
        f"results/{model_type}_generation.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved to: {output_file}"
    )


if __name__ == "__main__":
    evaluate()
