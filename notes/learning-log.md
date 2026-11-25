# Learning log

One dated entry per session. Format in AGENTS.md section 7.

## 2026-11-11/12 — Ch01: Setup + Transformer models
- **Did:** Set up repo, AGENTS.md and the Colab extension in Antigravity. Ran Ch1 pipeline demos on a CPU runtime: sentiment, zero-shot, text generation, fill-mask, NER, QA/summary/translation via a chat model, bias.
- **Broke:**
  - Colab picker closed on click → Developer: Reload Window.
  - Colab secrets and `drive.mount()` fail through the extension → hidden `getpass` token prompt; persist to the Hub (private repos).
  - Text-generation default is now SmolLM3-3B (~12 GB fp32), too big for CPU → name small models explicitly.
  - transformers v5 removed the question-answering/summarization/translation pipelines → one instruct model (Qwen2.5-0.5B-Instruct) prompted per task.
- **Learned:**
  - `pipeline` = tokenizer → model (logits) → post-processing (softmax → label/score).
  - A high softmax score is confidence, not correctness: "Great, another Monday." → POSITIVE 0.9997.
  - Encoders (BERT: fill-mask, NER) read both directions; decoders (SmolLM, Qwen) generate left to right; encoder-decoders (BART) map text to text.
  - Memory ≈ params × bytes/param (67M × 4 B ≈ 268 MB).
  - Cased models use capitalization: lowercase "spanish" → `span` + `##ish`, only half tagged.
  - Pretrained models inherit training-data bias (man → lawyer/doctor, woman → nurse/maid), and fine-tuning starts from those weights.

## 2026-11-12/16 — Ch02: Using Transformers
- **Did:** Ran sections 2.2–2.6 on a CPU runtime, mirroring the course pages cell by cell, with extra "👀 Look" cells to show hidden results. Loaded 2.8 (TGI/vLLM/llama.cpp) as reading plus CPU sampling and KV-cache demos (not yet run). Pushed a first private model repo (`my-awesome-model`).
- **Broke:**
  - Colab kernel picker did nothing again → Developer: Reload Window.
  - `SyntaxError: '(' was never closed` → the editor ran an unsaved half-typed cell; re-typed.
  - The course page's sentence-pair output is wrong: `tokenizer(a, b)` gives ONE pair `[CLS] A [SEP] B [SEP]` with token_type_ids 0/1, not two lists.
  - `torch.tensor` on ragged lists → `ValueError` (expected; padding fixes it). 1-D input to the model → `RuntimeError` in v5 (`IndexError` in the course).
  - v5 differences: tokenizer returns `token_type_ids` for DistilBERT; `decode` no longer cleans spaces (`I ' m`); loading a body-only `AutoModel` from a classifier checkpoint prints a LOAD REPORT with UNEXPECTED head keys (harmless).
- **Learned:**
  - Pipeline by hand: tokenizer → `input_ids` + `attention_mask` → body gives hidden states (batch, seq, 768) → head gives logits (batch, labels) → softmax → `id2label`.
  - `Auto*` classes read `config.json` to pick the architecture; `save_pretrained` = `config.json` + `model.safetensors`.
  - Subword tokenization (`Trans` + `##former`) balances vocabulary size, unknown words and sequence length.
  - Models need rectangular 2-D batches: padding makes them rectangular; the attention mask sets [PAD] scores to −∞ before softmax so padding has zero weight (padded row logits then match the unpadded run).
  - Padding never truncates: `padding="max_length", max_length=8` leaves a 16-token row at 16.

## 2026-11-16/17 — Ch03: Processing the data (3.2), Trainer API prepared (3.3)
- **Did:** Connected a T4 GPU server. Ran 3.2 on MRPC: one manual training step, `load_dataset`, `Dataset.map(batched=True)`, dynamic padding with `DataCollatorWithPadding`; solved the exercises (labels of train[15]/validation[87], single vs pair tokenization, SST-2 preprocessing). Loaded 3.3 (Trainer, evaluation, fp16 / gradient accumulation / LR schedules) with Hub checkpointing via `hub_args`; not run yet.
- **Broke:**
  - `load_dataset("glue", "mrpc")` → `HfUriError` in datasets 4.x; use `"nyu-mll/glue"`.
  - `tokenizer(raw_datasets["train"]["sentence1"])` → `ValueError`: datasets 4.x returns a lazy `Column`; wrap in `list(...)`.
  - Colab free GPU quota ran out ("no resources") → waiting for it to reset before training 3.3.
- **Learned:**
  - When `labels` are in the batch, the model returns a cross-entropy loss; a random head gives ~ln 2 ≈ 0.69 on 2 classes, more if it is confidently wrong (got 0.94).
  - `tokenizer(a, b)` builds `[CLS] A [SEP] B [SEP]` with token_type_ids 0/1; a pair is len(A) + len(B) − 1 tokens (one shared [CLS]).
  - Don't pad at `map` time: the collator pads each batch to its own longest sample, which removes most [PAD] compute.
  - The preprocessing recipe is the same for any text-classification dataset; only the tokenize function changes (one column for SST-2, two for MRPC).
