# llm-workbench

This is my hands-on workbench for how language models work under the hood. The method is simple: take a library call that hides everything, rebuild it by hand, and look at the numbers on the way. It covers inference and fine-tuning so far. Pretraining is next and is not here yet.

The notebooks follow the [Hugging Face LLM Course](https://huggingface.co/learn/llm-course). The course was written for transformers v4; here everything runs on transformers v5 and datasets 4.x, so the code that broke has been adapted and the cell says what changed. Cells marked 👀 are not in the course. They print what the library normally hides.

## Notebooks

[`ch01_transformer_models.ipynb`](notebooks/ch01_transformer_models.ipynb) runs pipelines for sentiment, zero-shot classification, generation, mask filling and NER. transformers v5 removed the question-answering, summarization and translation pipelines, so one small chat model (Qwen2.5-0.5B-Instruct) does those three from prompts. The last demo shows the bias a pretrained model brings along.

[`ch02_using_transformers.ipynb`](notebooks/ch02_using_transformers.ipynb) takes the pipeline apart: text → token IDs and attention mask → hidden states → logits → softmax. It also shows why batches need padding, and why padding without a mask changes the answer. Section 2.8 (serving with TGI, vLLM and llama.cpp) is there to read; its sampling and KV-cache cells have not been run yet.

[`ch03_fine_tuning.ipynb`](notebooks/ch03_fine_tuning.ipynb) fine-tunes `bert-base-uncased` on MRPC: given two sentences, decide whether they mean the same thing. It starts with one training step by hand, then uses the `Trainer`, then writes the same loop in plain PyTorch and again with Accelerate. It ends with learning curves and early stopping.

## Results

All runs are `bert-base-uncased` on the MRPC validation set (408 pairs), trained for 3 epochs at batch size 8.

| Run | Accuracy | F1 |
|---|---|---|
| Baseline: always answer "paraphrase" | 0.684 | 0.812 |
| `Trainer`, defaults | 0.838 | 0.889 |
| `Trainer`, defaults, evaluating every epoch | 0.860 | 0.901 |
| `Trainer`, fp16 | 0.875 | 0.914 |
| PyTorch loop, lr 5e-5 | 0.841 | 0.888 |
| Accelerate loop, lr 3e-5 | 0.885 | 0.919 |

Do not read this as a ranking. Each row is one run, and with 408 examples the standard error of an accuracy around 0.85 is about 1.8 points. Rows two and three train with the same hyperparameters and still land 2.2 points apart.

The more useful result is the overfitting. In the per-epoch run the validation loss goes 0.360 → 0.500 → 0.733 while accuracy goes 0.836 → 0.858 → 0.860. The model keeps getting slightly more answers right and becomes far more confident about the ones it gets wrong. In a separate 10-epoch run, early stopping on validation loss ended training at step 800 of 4,590, with the best checkpoint at step 500.

## Running it

The notebooks run on Google Colab. Chapters 1 and 2 are fine on CPU; chapter 3 wants a T4. The first cell installs the dependencies and logs in to the Hugging Face Hub, using a Colab secret called `HF_TOKEN_COURSE` or a hidden prompt, so no token is stored in the notebook. Use a token with write access, because chapters 2 and 3 push models to private repos on your account.

[`notes/learning-log.md`](notes/learning-log.md) records what was done, what broke and what it taught.

## Credits

Much of the code and some of the text in the notebooks come from the [Hugging Face LLM Course](https://github.com/huggingface/course), licensed under [Apache 2.0](https://github.com/huggingface/course/blob/main/LICENSE), and have been modified here.
