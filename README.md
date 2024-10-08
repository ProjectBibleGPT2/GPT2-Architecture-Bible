# GPT2 Architecture Bible

This repo is used for the training of [Inoob/Bible_GPT2](https://huggingface.co/Inoob/Bible_GPT2).

## Files

### Bible_KJV.txt

Full KJV Bible Corpus from [Bible Hub](https://biblehub.com/)'s [Open Bible Project](https://openbible.com/downloads.htm)

### Bible_NAB.txt

Full NAB Bible Corpus from [Bible Hub](https://biblehub.com/)'s [Open Bible Project](https://openbible.com/downloads.htm)

### KJV_NAB

Full KJV *and* NAB Bible Corpus from [Bible Hub](https://biblehub.com/)'s [Open Bible Project](https://openbible.com/downloads.htm)

### process_text.py

Used to clear the format:

1. Clears all TABs

2. Clears all "[" and "]"s

### randmnize_null_gpt2.py

Given a model and a text file for custom tokenizer, it clears all pretrained weights and randomnize the weights.

Outputs the randomnized model and custom tokenizer fit from given corpus

### train.py

Used to train the model, based on my [kaggle notebook](https://www.kaggle.com/code/ivanhe123/bblgt2).

## Usage

For cuda:

```
python -m pip install -r requirements_cuda.txt
```

For CPU

```
python -m pip install -r requirements_cpu.txt
```
then just run train.py
