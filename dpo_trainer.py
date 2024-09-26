import os
import gc
import json  # Added for handling JSON files
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset, Features, Value  # Imported Features and Value for schema validation
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import DPOTrainer, DPOConfig
import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='dpo_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_args():
    parser = argparse.ArgumentParser(description="DPO Fine-tuning for Language Models")
    parser.add_argument(
        "--base_model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--new_model",
        default="DPO-Mistral-7B-Instruct-v0.3",
        help="Name for the fine-tuned model"
    )
    parser.add_argument(
        "--dataset_name",
        default=None,  # Made default None to allow usage without specifying if data_file is used
        help="Name of the dataset to use for fine-tuning (e.g., 'Intel/orca_dpo_pairs', 'Anthropic/hh-rlhf')"
    )
    parser.add_argument(
        "--data_file",
        default=None,  # New argument for local dataset file
        help="Path to the local dataset file in JSON format"
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Subset of the dataset to use (e.g., 'helpful-base' for 'Anthropic/hh-rlhf')"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to use from the dataset"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum number of training steps"
    )
    return parser.parse_args()

def infer_model_type(base_model_name):
    """
    Infers the model type based on the base_model_name.

    Args:
        base_model_name (str): The name of the base model.

    Returns:
        str: 'mistral', 'llama2', or 'llama3'.

    Raises:
        ValueError: If the model type cannot be inferred.
    """
    base_model_lower = base_model_name.lower()
    if 'mistral' in base_model_lower:
        return 'mistral'
    elif 'llama-2' in base_model_lower or 'llama2' in base_model_lower:
        return 'llama2'
    elif 'llama' in base_model_lower:
        return 'llama3'
    else:
        raise ValueError(f"Unable to infer model type from base_model name: {base_model_name}. Please ensure it contains 'mistral', 'llama-2', or 'llama'.")

def get_prompt_template(model_type):
    if model_type == "mistral":
        return '''{% if messages[0]['role'] == 'system' %}
{% set loop_messages = messages[1:] %}
{% set system_message = messages[0]['content'] %}
{% else %}
{% set loop_messages = messages %}
{% set system_message = false %}
{% endif %}
{% if system_message != false %}
System: {{ system_message }}

{% endif %}
{% for message in loop_messages %}
{% if message['role'] == 'user' %}
Human: {{ message['content'] }}

{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}

{% endif %}
{% endfor %}
{% if add_generation_prompt %}
Assistant: {% endif %}'''
    elif model_type == "llama2":
        return '''{% if messages[0]['role'] == 'system' %}
    {% set system_message = '<<SYS>>\n' + messages[0]['content'] | trim + '\n<</SYS>>\n\n' %}
    {% set messages = messages[1:] %}
{% else %}
    {% set system_message = '' %}
{% endif %}

{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if loop.index0 == 0 %}
        {% set content = system_message + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' ' + content | trim + ' ' + eos_token }}
    {% endif %}
{% endfor %}'''
    elif model_type == "llama3":
        return '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant.<|eot_id|>
{% for message in messages %}
{% if message['role'] == 'user' %}
<|start_header_id|>user<|end_header_id|>
{{ message['content'] }}
<|eot_id|>
{% elif message['role'] == 'assistant' %}
<|start_header_id|>assistant<|end_header_id|>
{{ message['content'] }}
<|eot_id|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>
{% endif %}'''

def load_models_and_tokenizer(base_model, model_type):
    logging.info(f"Loading base model: {base_model} with inferred model type: {model_type}")
    print(f"Loading base model: {base_model} with inferred model type: {model_type}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Ensure the tokenizer has a pad_token
    if tokenizer.pad_token is None:
        logging.info("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        print("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side and pad_token_id based on model type
    if model_type == "mistral":
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_type == "llama2":
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token  # Example pad_token_id for Llama2; adjust if different
    elif model_type == "llama3":
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = 128001  # As per the evaluation script

    prompt_template = get_prompt_template(model_type)
    tokenizer.chat_template = prompt_template

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # For LLaMA2 and LLaMA3 models, ensure the tokenizer recognizes the special tokens
    if model_type in ["llama2", "llama3"]:
        if model_type == "llama2":
            special_tokens = {
                "additional_special_tokens": ["<<SYS>>", "<</SYS>>", "[INST]", "[/INST]"]
            }
        elif model_type == "llama3":
            special_tokens = {
                "additional_special_tokens": ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
            }
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        ref_model.resize_token_embeddings(len(tokenizer))

    return model, ref_model, tokenizer

def prepare_orca_dataset(dataset_name, num_samples, tokenizer, model_type):
    """
    Prepares the Orca dataset for DPO training.

    Args:
        dataset_name (str): Name of the Orca dataset.
        num_samples (int): Number of samples to select.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to apply.
        model_type (str): Type of the model ('mistral', 'llama2', or 'llama3').

    Returns:
        datasets.Dataset: Reformatted Orca dataset.
    """
    logging.info(f"Loading Orca dataset: {dataset_name}")
    print(f"Loading Orca dataset: {dataset_name}")

    dataset = load_dataset(dataset_name)['train']
    dataset = dataset.shuffle(seed=17345).select(range(min(num_samples, len(dataset))))

    prompt_template = tokenizer.chat_template

    def dataset_format(example):
        system_content = example.get('system', '')
        if system_content:
            message = {"role": "system", "content": system_content}
            system = tokenizer.apply_chat_template([message], tokenize=False)
        else:
            system = ""
        user_message = {"role": "user", "content": example['question']}
        prompt = tokenizer.apply_chat_template([user_message], tokenize=False, add_generation_prompt=True)
        chosen = example['chosen'].strip() + " <|eot_id|>\n"
        rejected = example['rejected'].strip() + " <|eot_id|>\n"
        return {
            "prompt": system + prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    original_columns = dataset.column_names

    formatted_dataset = dataset.map(
        dataset_format,
        remove_columns=original_columns,
        num_proc=os.cpu_count(),
    )

    return formatted_dataset

def prepare_hh_rlhf_dataset(num_samples, tokenizer, model_type, subset=None):
    """
    Prepares the Anthropic/hh-rlhf dataset for DPO training.

    Args:
        num_samples (int): Number of samples to select.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to apply.
        model_type (str): Type of the model ('mistral', 'llama2', or 'llama3').
        subset (str): Subset of the dataset to load (e.g., 'helpful-base').

    Returns:
        datasets.Dataset: Reformatted hh-rlhf dataset.
    """
    subset = subset or "helpful-base"
    logging.info(f"Loading Anthropic/hh-rlhf dataset subset: {subset} from Hugging Face")
    print(f"Loading Anthropic/hh-rlhf dataset subset: {subset} from Hugging Face")

    dataset = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split='train')
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    default_system_prompt = "You are a helpful and harmless AI assistant."

    prompt_template = tokenizer.chat_template

    def reformat_example(example):
        """
        Reformats each example to include 'prompt', 'chosen', and 'rejected' fields.

        Assumes that each example has 'chosen' and 'rejected' fields.
        """
        user_prompt = example.get('prompt', '[User prompt here]')
        if model_type == "mistral":
            prompt = f"{default_system_prompt}\n\nHuman: {user_prompt}\nAssistant:"
        elif model_type == "llama2":
            prompt = f"<<SYS>>\n{default_system_prompt}\n<</SYS>>\n\n[INST] {user_prompt} [/INST]"
        elif model_type == "llama3":
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{default_system_prompt}<|eot_id|>\n\n<|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"

        chosen_response = example['chosen'].strip() + " <|eot_id|>\n"
        rejected_response = example['rejected'].strip() + " <|eot_id|>\n"

        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }

    formatted_dataset = dataset.map(
        reformat_example,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
    )

    return formatted_dataset

def prepare_local_file_dataset(data_file, tokenizer, model_type):
    """
    Prepares the dataset from a local JSON file for DPO training, including schema validation.

    Args:
        data_file (str): Path to the JSON file containing the dataset.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to apply.
        model_type (str): Type of the model ('mistral', 'llama2', or 'llama3').

    Returns:
        datasets.Dataset: Reformatted and validated dataset.
    """
    logging.info(f"Loading dataset from local file: {data_file}")
    print(f"Loading dataset from local file: {data_file}")

    with open(data_file, 'r', encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON file: {e}")
            print(f"Error: Failed to decode JSON file: {e}")
            raise

    # Ensure data is a list of dicts with 'prompt', 'chosen', 'rejected'
    if not isinstance(data, list):
        logging.error("Data file must contain a JSON array of examples.")
        print("Error: Data file must contain a JSON array of examples.")
        raise ValueError("Data file must contain a JSON array of examples.")

    # Define the expected schema using datasets.Features
    expected_features = Features({
        "prompt": Value("string"),
        "chosen": Value("string"),
        "rejected": Value("string"),
    })

    # Validate each example and log any discrepancies
    for idx, example in enumerate(data):
        if not isinstance(example, dict):
            logging.error(f"Example at index {idx} is not a dictionary.")
            print(f"Error: Example at index {idx} is not a dictionary.")
            raise ValueError(f"Example at index {idx} is not a dictionary.")
        for key in ['prompt', 'chosen', 'rejected']:
            if key not in example:
                logging.error(f"Example at index {idx} is missing the '{key}' field.")
                print(f"Error: Example at index {idx} is missing the '{key}' field.")
                raise ValueError(f"Example at index {idx} is missing the '{key}' field.")
            if not isinstance(example[key], str):
                print(example)
                logging.error(f"The '{key}' field in example at index {idx} is not a string.")
                print(f"Error: The '{key}' field in example at index {idx} is not a string.")
                raise ValueError(f"The '{key}' field in example at index {idx} is not a string.")

    try:
        # Create a datasets.Dataset from the data with schema validation
        dataset = Dataset.from_list(data, features=expected_features)
    except Exception as e:
        logging.error(f"Failed to create dataset with the expected schema: {e}")
        print(f"Error: Failed to create dataset with the expected schema: {e}")
        raise

    # Optionally, apply any necessary formatting similar to other datasets
    def format_example(example):
        # Depending on the model type, you might need to adjust the prompt
        if model_type == "mistral":
            prompt = f"{example['prompt']}\nAssistant:"
        elif model_type == "llama2":
            prompt = f"<<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n[INST] {example['prompt']} [/INST]"
        elif model_type == "llama3":
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant.<|eot_id|>\n\n<|start_header_id|>user<|end_header_id|>{example['prompt']}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"

        return {
            "prompt": prompt,
            "chosen": example['chosen'].strip() + " <|eot_id|>\n",
            "rejected": example['rejected'].strip() + " <|eot_id|>\n",
        }

    formatted_dataset = dataset.map(
        format_example,
        num_proc=os.cpu_count(),
    )

    return formatted_dataset

def prepare_dataset(dataset_name, data_file, num_samples, tokenizer, model_type, subset=None):
    """
    Prepares the dataset based on the dataset name or local file.

    Args:
        dataset_name (str): Name of the dataset to use.
        data_file (str): Path to the local dataset file.
        num_samples (int): Number of samples to select.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to apply.
        model_type (str): Type of the model ('mistral', 'llama2', or 'llama3').
        subset (str): Subset of the dataset to load (if applicable).

    Returns:
        datasets.Dataset: Reformatted dataset.
    """
    if data_file:
        logging.info(f"Preparing dataset from local file: {data_file}")
        print(f"Preparing dataset from local file: {data_file}")
        dataset = prepare_local_file_dataset(data_file, tokenizer, model_type)
        # If num_samples is specified and less than the dataset size, select a subset
        if num_samples and num_samples < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
    else:
        logging.info(f"Preparing dataset: {dataset_name} for model type: {model_type}")
        print(f"Preparing dataset: {dataset_name} for model type: {model_type}")

        if dataset_name.lower() == "anthropic/hh-rlhf":
            dataset = prepare_hh_rlhf_dataset(num_samples, tokenizer, model_type, subset=subset)
        else:
            # Assume other datasets follow a similar structure to 'orca'
            dataset = prepare_orca_dataset(dataset_name, num_samples, tokenizer, model_type)

    return dataset

def setup_dpo_trainer(model, ref_model, dataset, tokenizer, new_model, max_steps):
    logging.info("Setting up DPO Trainer")
    print("Setting up DPO Trainer")

    # Unified target_modules for all models
    target_modules = ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

    training_args = DPOConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        max_steps=max_steps,
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
        output_dir=new_model,
        optim="paged_adamw_32bit",
        warmup_steps=100,
        bf16=True,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.1,
        max_prompt_length=1024,
        max_length=2048,
        force_use_ref_model=True
    )

    return dpo_trainer

def main():
    args = setup_args()

    # Validate arguments
    if args.data_file and args.dataset_name:
        logging.error("Both --data_file and --dataset_name are provided. Please specify only one.")
        print("Error: Provide either --data_file or --dataset_name, not both.")
        exit(1)
    if not args.data_file and not args.dataset_name:
        logging.error("No dataset source provided. Please specify --data_file or --dataset_name.")
        print("Error: You must provide either --data_file or --dataset_name.")
        exit(1)

    try:
        model_type = infer_model_type(args.base_model)
    except ValueError as e:
        logging.error(str(e))
        print(str(e))
        exit(1)

    start_time = datetime.now()
    logging.info(f"Starting DPO fine-tuning at {start_time} for model type: {model_type}")
    print(f"Starting DPO fine-tuning at {start_time} for model type: {model_type}")

    model, ref_model, tokenizer = load_models_and_tokenizer(args.base_model, model_type)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = prepare_dataset(
        dataset_name=args.dataset_name,
        data_file=args.data_file,
        num_samples=args.num_samples,
        tokenizer=tokenizer,
        model_type=model_type,
        subset=args.subset  # Optional subset parameter
    )
    dpo_trainer = setup_dpo_trainer(
        model=model,
        ref_model=ref_model,
        dataset=dataset,
        tokenizer=tokenizer,
        new_model=args.new_model,
        max_steps=args.max_steps
    )

    logging.info("Starting training")
    print("Starting training")
    dpo_trainer.train()

    logging.info("Saving fine-tuned model")
    print("Saving fine-tuned model")
    dpo_trainer.model.save_pretrained(f"{args.new_model}_final")
    tokenizer.save_pretrained(f"{args.new_model}_final")

    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"DPO fine-tuning completed at {end_time}. Total duration: {duration}")
    print(f"DPO fine-tuning completed at {end_time}. Total duration: {duration}")

if __name__ == "__main__":
    main()
