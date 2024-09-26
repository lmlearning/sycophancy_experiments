import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

def setup_args():
    """
    Sets up command-line arguments for the inference script.
    """
    parser = argparse.ArgumentParser(description="Inference for Llama3 or Mistral (original or fine-tuned)")
    parser.add_argument("--model_path", required=True, help="Path to the model (original or fine-tuned)")
    parser.add_argument("--model_type", required=True, choices=['llama', 'mistral'], help="Type of the model: 'llama' or 'mistral'")
    parser.add_argument("--prompt", required=True, help="Prompt for inference")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top p for nucleus sampling")
    return parser.parse_args()

def load_model_and_tokenizer(model_path):
    """
    Loads the pre-trained model and tokenizer from the specified path.

    Args:
        model_path (str): Path to the model directory.

    Returns:
        model (AutoModelForCausalLM): Loaded language model.
        tokenizer (AutoTokenizer): Loaded tokenizer.
    """
    print(f"Loading model and tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    return model, tokenizer

def format_prompt_llama(input_text):
    """
    Formats the list of messages according to Llama 3's expected Instruct Chat template.

    Args:
        messages (list): A list of message dictionaries, each containing 'role' and 'content'.
                         Example:
                         [
                             {"role": "system", "content": "You are a helpful AI assistant."},
                             {"role": "user", "content": "What is the capital of France?"},
                             {"role": "assistant", "content": "Paris is the capital of France."},
                             {"role": "user", "content": "Thank you!"}
                         ]

    Returns:
        str: A formatted prompt string adhering to Llama 3's Instruct Chat format.
    """
    # Initialize the prompt with the beginning token
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": input_text}
    ]
    prompt = "<|begin_of_text|>"

    for message in messages:
        role = message['role']
        content = message['content']

        # Add the role header
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n"

        # Add the message content
        prompt += f"{content}<|eot_id|>\n\n"

    # Append the assistant header to prompt the model for a response
    prompt += "<|start_header_id|>assistant<|end_header_id|>"

    return prompt

def format_prompt_mistral(tokenizer, prompt):
    """
    Formats the prompt according to Mistral's expected template using the tokenizer's apply_chat_template method.

    Args:
        tokenizer (AutoTokenizer): The tokenizer for the Mistral model.
        prompt (str): The user-provided prompt.

    Returns:
        torch.Tensor: Tokenized input IDs suitable for model inference.
    """
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply the chat template to get tokenized inputs
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    input_ids = encodeds['input_ids']
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    return input_ids

def generate_response_llama(pipeline, prompt, max_length, temperature, top_p):
    """
    Generates a response using the Llama 3 model via the TextGenerationPipeline.

    Args:
        pipeline (TextGenerationPipeline): The text generation pipeline.
        prompt (str): The formatted prompt string.
        max_length (int): Maximum length of generated text.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.

    Returns:
        str: Generated response text.
    """
    print("Generating response with Llama 3...")
    sequences = pipeline(
        prompt,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1,
        max_length=max_length,
    )
    return sequences[0]['generated_text']

def generate_response_mistral(model, tokenizer, input_ids, max_length, temperature, top_p):
    """
    Generates a response using the Mistral model by directly invoking model.generate.

    Args:
        model (AutoModelForCausalLM): The Mistral model.
        tokenizer (AutoTokenizer): The Mistral tokenizer.
        input_ids (torch.Tensor): Tokenized input IDs.
        max_length (int): Maximum length of generated text.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.

    Returns:
        str: Generated response text.
    """
    print("Generating response with Mistral...")
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Decode the generated tokens, skipping special tokens
    decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Extract the assistant's response by removing the input prompt
    # This assumes that the model appends the response after the prompt
    # Adjust slicing as necessary based on model's output format
    response = decoded_responses[0][len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
    return response.strip()

def main():
    args = setup_args()
    
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    if args.model_type.lower() == 'llama':
        # Initialize the TextGenerationPipeline for Llama
        pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
        
        # Format the prompt using Llama's chat template
        formatted_prompt = format_prompt_llama(args.prompt)
        
        # Generate the response
        response = generate_response_llama(
            pipeline,
            formatted_prompt,
            args.max_length,
            args.temperature,
            args.top_p
        )
    
    elif args.model_type.lower() == 'mistral':
        # Format the prompt using Mistral's chat template
        input_ids = format_prompt_mistral(tokenizer, args.prompt)
        
        # Generate the response
        response = generate_response_mistral(
            model,
            tokenizer,
            input_ids,
            args.temperature,
            args.top_p
        )
    
    else:
        raise ValueError("Unsupported model type. Choose 'llama' or 'mistral'.")
    
    print("\nGenerated Response:")
    print(response)

if __name__ == "__main__":
    main()
