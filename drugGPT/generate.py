import os
import torch
import logging

from transformers import AutoTokenizer, GPT2LMHeadModel
config = {
            "generation_kwargs": {
                "do_sample": True,
                "top_k": 9,
                "max_length": 1024,
                "top_p": 0.9,
                "num_return_sequences": 10
            },
            "max_retries": 30  # Max retry limit to avoid infinite loops
        }
def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer from HuggingFace."""
    logging.info(f"Loading model and tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        # Move the model to CUDA if available
        if torch.cuda.is_available():
            logging.info("Moving model to CUDA device.")
            model = model.to("cuda")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {e}")
        raise RuntimeError(f"Failed to load model and tokenizer: {e}")

os.environ['INFRA_PROVIDER'] = '叫什么无所谓，他只是判断这个环境变量是不是存在'
model_name = f"./druggen"
model, tokenizer = load_model_and_tokenizer(model_name)

def GPT(sequence):
    prompt = f"<|startoftext|><P>{sequence}<L>"
    encoded_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    generation_kwargs = config["generation_kwargs"]
    generation_kwargs["bos_token_id"] = tokenizer.bos_token_id
    generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    sample_outputs = model.generate(encoded_prompt, **generation_kwargs)
    for sample_output in sample_outputs:
        output_decode = tokenizer.decode(sample_output, skip_special_tokens=False)
        generated_smiles = output_decode.split("<L>")[1].split("<|endoftext|>")[0]
    return generated_smiles
