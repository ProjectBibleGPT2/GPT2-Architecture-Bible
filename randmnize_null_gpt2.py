from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tokenizers import (
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import torch
import os
def randomize(input_model):
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
    trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
    tokenizer.model = models.BPE()
    tokenizer.train(["cleared.txt"], trainer=trainer)
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
