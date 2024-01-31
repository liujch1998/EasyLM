gcloud alpha compute tpus tpu-vm ssh jiachengl-v2-8 --zone=us-central1-f --project=ai2-tpu --worker=all --command="cd n-tulu-ppo-jax; git pull; export WANDB_API_KEY='a46519994b4614615d5ce4aa8742ef19685a7cae'; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; python3 -m EasyLM.models.llama.llama_train_ppo \
    --mesh_dim='1,8,1' \
    --load_llama_config_policy='debug' \
    --load_llama_config_reward='debug' \
    --load_checkpoint_policy='' \
    --load_checkpoint_reward='' \
    --tokenizer.vocab_file='gs://hamishi-dev/easylm/llama/tokenizer.model' \
    --tokenizer.add_bos_token=True \
    --train_dataset.type='hf_prompt' \
    --train_dataset.text_processor.fields='[instruction]' \
    --train_dataset.hf_prompt_dataset.seq_length=64 \
    --max_continuation_len=8 \
    --train_dataset.hf_prompt_dataset.batch_size=8 \
    --mini_batch_size=8 \
    --train_dataset.hf_prompt_dataset.num_workers=32 \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.0 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=False \
    --logger.entity='liujch1998' \
    --logger.project='n-Tulu-PPO-Jax' \
    --logger.prefix='debug_tpu' \
    --logger.prefix_to_id=True \
    --logger.wandb_dir='wandb' \
    --logger.output_dir='/home/jiachengl/n-tulu-ppo-jax/runs/' \
    --use_tpu=True \
    --ppo_epochs=1 \
    --save_model_freq=0 \
    --max_steps_per_epoch=1 \
    --generate_only=False \
    &> ~/n-tulu-ppo-jax/all.log &"
