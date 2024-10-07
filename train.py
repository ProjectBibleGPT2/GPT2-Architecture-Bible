# from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, get_cosine_with_hard_restarts_schedule_with_warmup,DataCollatorForLanguageModeling
import torch
import os
import math
from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrainerCallback

class AdjustLrCallback(TrainerCallback):
    def __init__(self, threshold, x,factor):
        self.threshold = threshold
        self.x = x
        self.losses = []
        self.factor = factor

    def on_log(self, args, state, control, **kwargs):
        current_loss = kwargs['logs'].get('loss')
        if current_loss is not None:
            self.losses.append(current_loss)
            if len(self.losses) > self.x:
                self.losses.pop(0)
                if max(self.losses) - min(self.losses) <= self.threshold:
                    for param_group in kwargs['optimizer'].param_groups:
                        param_group['lr'] *= 1-self.factor
                    

import process_text, randmnize_null_gpt2 
from datasets import load_dataset

process_text.clean("/kaggle/working/GPT2-Architecture-Bible/Bible_KJV.txt", "/kaggle/working/cleared.txt")
#process_text.clean("/kaggle/working/GPT2-Architecture-Bible/Bible_NAB.txt", "/kaggle/working/cleared_NAB.txt")

model, tokenizer = randmnize_null_gpt2.randomize("openai-community/gpt2-large", "/kaggle/working/cleared.txt")
output_dir = "./trained_gpt2"
# Set the padding token 
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset('text', data_files={'train': '/kaggle/working/cleared.txt'})
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Create a data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=15000,
    save_total_limit=2,
    fp16=True,
)
def startnewtrain(model):
    # Initialize the AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=6e-5)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        data_collator=data_collator,
        optimizers=(optimizer, None),  # Pass the optimizer and scheduler
        callbacks=[AdjustLrCallback(threshold=0.1, x=2, factor=0.5)]
    )

    # Move model to GPU
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer.train()
    # Save the model
    model.save_pretrained(f"{output_dir}{x}")
    return model.cpu()
models=[model]+[None for x in range(2)]
"""models[1] = startnewtrain(models[0])
models[2] = startnewtrain(models[1])"""
for x in range(3-1):
    models[x+1] = startnewtrain(models[x])

    print(f"Model saved to {output_dir}{x}")
tokenizer.save_pretrained(output_dir)
