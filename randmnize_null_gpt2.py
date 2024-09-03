from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

model_name = "Inoob/Null-GPT2-Large"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Reset parameters
for name, param in model.named_parameters():
    if param.dim() > 1:
        torch.nn.init.xavier_uniform_(param)
    else:
        torch.nn.init.zeros_(param)



# Save the reset model and tokenizer
output_dir = "./gpt2-large-architecture"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Reset model and tokenizer saved to {output_dir}")