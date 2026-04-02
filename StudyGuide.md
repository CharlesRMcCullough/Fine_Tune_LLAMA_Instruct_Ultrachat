# LLAMA + UltraChat Fine-Tuning Study Guide

## 1. Dataset
- `load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")`
- Shuffle and select subset: `.shuffle(seed=0).select(range(10_000))`
- Format raw messages to model prompt text using `template_tokenizer.apply_chat_template` and store as `text` field.
- Verify sample output with `dataset[0]["text"]`.

## 2. Model and tokenizer setup
- Base model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- `AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)`
- Set `pad_token` and `padding_size="left"`.

## 3. Quantization (Q-LoRA) config via BitsAndBytes
- Use `BitsAndBytesConfig`:
  - `load_in_4bit=True`
  - `bnb_4bit_quant_type="nf4"`
  - `bnb_4bit_compute_dtype='float16'`
  - `bnb_4bit_use_double_quant=True`
- Load model with `quantization_config=bnb_config` and `device_map="auto"`.
- Set `model.config.use_cache=False` before training.

## 4. LORA + PEFT setup
- `LoraConfig` parameters:
  - `r=64`, `lora_alpha=32`, `lora_dropout=0.1`
  - `bias='none'`, `task_type='CAUSAL_LM'`
  - `target_modules=[...]` for transformer attention matrices
- `prepare_model_for_kbit_training(model)` before `get_peft_model`.
- `model = get_peft_model(model, peft_config)` to apply PEFT adapter.

## 5. Trainer config and error patterns
- `SFTConfig` for `trl` training arguments:
  - `output_dir`, `per_device_train_batch_size`, `gradient_accumulation_steps`.
  - `optim='paged_adamw_32bit'`, `learning_rate`, `lr_scheduler_type='cosine'`, `num_train_epochs`, `logging_steps`.
  - Mixed precision: `fp16=True`, `gradient_checkpointing=True`.
  - Prompt options: `dataset_text_field='text'`, `max_length=512`.

- `SFTTrainer` constructor for this trl version expects:
  - `model` (base model), `args` (`SFTConfig`), `train_dataset`, `peft_config`, `processing_class=tokenizer`
  - Do NOT pass `dataset_text_field` directly to `SFTTrainer` (set in config), else `TypeError`.

- If using `PeftModel` from previous adapter:
  - Call `merge_and_unload` on adapter model.
  - Save merged base model and reload for new training.

## 6. Save/Load and inference pattern
- After training: `trainer.model.save_pretrained("TinyLlama-1.1B-qlora")`
- Load with `AutoPeftModelForCausalLM.from_pretrained(..., device_map='auto')`
- Merge adapter: `merged_model = model.merge_and_unload()`
- Run generation with `pipeline(task='text-generation', model=merged_model, tokenizer=tokenizer)`.

## 7. Troubleshooting notes
- Check `trl` version and API matching (`SFTTrainer` signature and `SFTConfig` fields).
- If `unexpected keyword argument 'dataset_text_field'` occurs, move that to `SFTConfig` and provide base model + peft config to trainer.
- If `PeftModel` with `peft_config` error, unify by merging/unloading adapter into base model first.

## 8. Further study
- `trl` SFT concepts: `SFTTrainer`, `SFTConfig`, `dataset_text_field`, `processing_class`.
- PEFT/LoRA internal mechanics and target module selection.
- 4-bit quantization theory (NF4, double quant, compute dtype).
- Hugging Face Dataset preprocessing and prompt engineering for chat data.
- A100/RTX usage, gradient checkpointing, and batch accumulation strategies.
