import os
import time
import json
import torch

# huggingface -> used to load the dataset and tokenizer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR = "output"
TRAIN_FILE = "data/train.json"

# true: QloRa Quantization, false: Lora
NEED_QUANT = True

def load_data():
    # Load the dataset from the specified JSON file
    dataset = load_dataset("json", data_files = TRAIN_FILE, split = "train")

    def convert(example):
        # Convert the example to the required format
        question = example["messages"][0]["content"]
        answer = example["messages"][1]["content"]

        return {
            "prompt": f"Question: {question}\nAnswer:",
            "completion": f" {answer}"
        }

    return dataset.map(convert)


def train():
    # 1. tokenizer
    # load the corresponding tokenizer for the model
    # trust_remote_code=True is required for Qwen model, otherwise it will report an error
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # set the padding token to the end-of-sequence token,
    # because the original tokenizer does not have a padding token
    tokenizer.pad_token = tokenizer.eos_token

    # 2. data set
    dataset = load_data()

    # 3. quantization configuration
    quantization_config = None
    if NEED_QUANT:
        # 4 bit quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,                        # use 4-bit quantization
            bnb_4bit_quant_type = "nf4",                # normalized float 4 quantization
            bnb_4bit_compute_dtype = torch.float16,     # compute use the bfloat16 data type
            bnb_4bit_use_double_quant = True,           # double float, let coefficients be quantized
        )

    # 4. model load
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config = quantization_config,

        dtype = (
            torch.float16
            if torch.cuda.is_available() else torch.float32
        ),

        device_map = "auto",
        trust_remote_code = True
    )

    # 5. lora configuration
    peft_config = LoraConfig(
        r = 16,                     # rank of the low-rank decomposition  4096 * 16 x 16 * 4096
        lora_alpha = 32,            #
        lora_dropout = 0.05,        # lora dropout rate

        # x W -> x （W + ΔW） = x W + x ΔW = x W + x BA
        # ΔW = BA
        # dim A : r x input_dim
        # dim B : output_dim x r
        # B A -> output_dim x input_dim
        target_modules = [          # where to put the lora module
            # attention Q projection, K projection, V projection
            "q_proj",               # Q = X Wq -> Q = X Wq + X Wq_lora
            "k_proj",               # K = X Wk -> K = X Wk + X Wk_lora
            "v_proj",               # V = X Wv -> V = X Wv + X Wv_lora

            # prediciton head O projection
            # after the Attention(Q, K, V) -> O
            "o_proj",               # O = X Wo -> O = X Wo + X Wo_lora

            # (MLP)   Feed Forward Network (FFN) projection, two parallel linear layers
            # 1. gate_proj - SiLu
            # 2. up_proj - linear
            # together -> down_proj - linear
            # gate projection, G = X Wg
            "gate_proj",            # Gated Linear Unit (GLU) projection

            # projection to higher dimension, U = X Wu
            "up_proj",              # U = X Wu -> U = X Wu + X Wu_lora
            # projection to lower dimension, D = X Wd
            "down_proj"             # D = X Wd -> D = X Wd + X Wd_lora
        ],

        bias = "none",              # not train the bias term of the model
        task_type = "CAUSAL_LM"     # the task type is causal language modeling
    )

    # 6. set output directory
    output_dir = os.path.join(
        OUTPUT_DIR,
        "lora" if not NEED_QUANT else "qlora",
    )
    os.makedirs(OUTPUT_DIR, exist_ok = True)

    # 4. training
    trainning_args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs = 3,               # total epochs for all training data
        per_device_train_batch_size = 2,    # batch size
        gradient_accumulation_steps = 4,    # how many steps to accumulate gradients before updating
        learning_rate = 2e-4,               # learning rate
        logging_steps = 1,                  # how many steps to log the trainning loss
        save_strategy = "no",               # do not save the middle checkpoint
        report_to = "none",                 # close the log tools, like wandb
        bf16 = False,                       # use float16 for training, if your GPU does not support bfloat16, please set it to False
        fp16 = False,   # use fp16 for training, if your GPU does not support fp16, please set it to False
        gradient_checkpointing = True,      # do not store the intermediate results

        # quantization and optimizer settings
        # use the paged_adamw_8bit optimizer for QLoRA,
        # # and use the adamw_torch optimizer for LoRA
        optim = (
            "paged_adamw_8bit" if NEED_QUANT else "adamw_torch"
        )
    )


    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        processing_class = tokenizer,   # here truelly runs the tokenizaiton
        peft_config = peft_config,
        args = trainning_args,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # start training
    start_time = time.perf_counter()
    trainer.train()
    training_time = time.perf_counter() - start_time

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    peak_memory = 0

    # transfer to GB
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3

    # save the training result to a JSON file
    result = {
        "experiment_name": "qlora" if NEED_QUANT else "lora",
        "training_time": training_time,
        "peak_memory_gb": peak_memory
    }

    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent = 2)
        print(result)

if __name__ == "__main__":
    train()
