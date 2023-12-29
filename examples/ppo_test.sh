python -m EasyLM.models.llama.llama_train_ppo \
    --mesh_dim='-1,1,1' \
    --dtype='bf16' \
    --num_epochs=1000 \
    --log_freq=1 \
    --save_model_freq=-1 \
    --save_milestone_freq=-1 \
    --load_llama_config='debug' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --tokenizer.vocab_file='tokenizer.model' \ # 'gs://hamishi-dev/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=1e-3 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.0 \
    --optimizer.accumulate_gradient_steps=8 \
    --train_dataset.type='preference_json_torch' \
    --train_dataset.text_processor.fields='[prompt],completion' \
    --train_dataset.json_torch_dataset.path='debug_pref_3.json' \
    --train_dataset.json_torch_dataset.seq_length=128 \
    --train_dataset.json_torch_dataset.batch_size=2 \
    --train_dataset.json_torch_dataset.num_workers=32 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=False \
    --logger.output_dir=tmp