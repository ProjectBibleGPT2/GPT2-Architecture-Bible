from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tokenizers import (
    models,
    decoders,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import torch
import os
def get_vocab_size(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    words = text.split()
    unique_words = set(words)
    return len(unique_words)
def randomize(input_model, file_path):
    model_name = input_model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    

    # Reset parameters
    for name, param in model.named_parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=get_vocab_size(file_path), special_tokens=["<|endoftext|>"])
    tokenizer.model = models.BPE()
    tokenizer.train([file_path], trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    return model, tokenizer
if __name__ == "__main__":
    model,tokenizer =randomize("Inoob/Null-GPT2-Large")
    output_dir = "./gpt2-large-architecture"
    # Save the reset model and tokenizer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Reset model and tokenizer saved to {output_dir}")
