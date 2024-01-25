python3 -m EasyLM.models.llama.llama_train_ppo \
    --mesh_dim='-1,1,1' \
    --load_llama_config_policy='7b' \
    --load_llama_config_reward='13b' \
    --load_checkpoint_policy='params::gs://hamishi-dev/easylm/llama2/tulu2_7b_fixed/263f4f758b194729b206d5adad2b50d7/streaming_params' \
    --load_checkpoint_reward='params::/net/nfs.cirrascale/allennlp/jiachengl/n-tulu-ppo-jax/ckpt/UltraRM-13b/streaming_params' \ # 'params::gs://hamishi-dev/easylm/llama2/tulu2_7b_fixed/263f4f758b194729b206d5adad2b50d7/streaming_params' \
    --tokenizer.vocab_file='gs://hamishi-dev/easylm/llama/tokenizer.model' \
    --tokenizer.add_bos_token=True \
    --train_dataset.type='hf_prompt' \
    --train_dataset.text_processor.fields='[instruction]' \
    --train_dataset.hf_prompt_dataset.seq_length=64 \
    --train_dataset.hf_prompt_dataset.batch_size=1 \
    --train_dataset.hf_prompt_dataset.num_workers=32 \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.init_lr=1e-5 \
    --optimizer.adamw_optimizer.lr=1e-5 \
    --optimizer.adamw_optimizer.end_lr=1e-5 \
    --optimizer.adamw_optimizer.warmup_ratio=0.0 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True \
    --logger.entity='liujch1998' \
    --logger.project='n-Tulu-PPO-Jax' \
    --logger.prefix='7b' \
    --logger.prefix_to_id=True \
    --logger.wandb_dir='wandb' \
    --logger.output_dir='/net/nfs.cirrascale/allennlp/jiachengl/n-tulu-ppo-jax/runs/' \
    --use_tpu=False \
    --mini_batch_size=1 \
    --max_continuation_len=16 \
    --max_steps_per_epoch=1
