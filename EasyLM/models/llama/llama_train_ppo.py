'''
Llama train script modified for PPO.
WIP!!!
'''
import pprint
import math
import time
import copy

from tqdm import tqdm, trange
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
import flax
import torch

from ...data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    global_norm, get_float_dtype_by_name, set_random_seed,
    get_weight_decay_mask, make_shard_and_gather_fns,
    with_sharding_constraint
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLMModule, FlaxLLaMAForCausalLM
)
from transformers import GenerationConfig


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,-1,1',
    dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint_policy='',
    load_checkpoint_reward='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    num_epochs=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),

    ministeps=4,
    kl_coef=0.2,
    whiten_rewards=False,
    gamma=1.0,
    lam=0.95,
    cliprange=0.2,
    cliprange_value=0.2,
    policy_loss_coef=1.0,
    value_loss_coef=0.1,
    reward_gain=1.0,
    reward_bias=0.0,
    max_new_tokens=32,
)


def convert_logits_to_logps(logits, labels, loss_mask):
    labels = labels[:, 1:]
    logits = logits[:, :-1, :]
    logps = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), axis=-1, indices=labels[:,:,None]).squeeze(-1)
    logps = jnp.concatenate([jnp.zeros_like(logps[:, :1]), logps], axis=-1)
    logps = logps * loss_mask
    return logps

def whiten(rewards, mask, shift_mean=True):
    rewards = rewards * mask
    mean = jnp.sum(rewards, axis=-1, keepdims=True) / jnp.sum(mask, axis=-1, keepdims=True)
    rewards = rewards - mean
    if shift_mean:
        rewards = rewards + jnp.mean(rewards, axis=-1, keepdims=True)
    std = jnp.sqrt(jnp.sum(rewards ** 2, axis=-1, keepdims=True) / jnp.sum(mask, axis=-1, keepdims=True))
    rewards = rewards / std
    return rewards

def reduce_mean(x, mask):
    return jnp.sum(x * mask) / jnp.sum(mask)


def ppo_loss(
    policy_model, value_model,
    policy_params, value_params,
    rng,
    input_ids, attention_mask, continuation_mask, old_continuations_logps, old_continuations_values, rewards,
):
    if FLAGS.whiten_rewards:
        rewards = whiten(rewards, continuation_mask, shift_mean=False)
    lastgaelam = 0
    advantages_reversed = []
    len = continuation_mask.shape[1]
    for t in reversed(range(len)):
        delta = rewards[:, t] + FLAGS.gamma * old_continuations_values[:, t + 1] - old_continuations_values[:, t]
        lastgaelam = delta + FLAGS.gamma * FLAGS.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = jnp.stack(advantages_reversed[::-1], axis=1)
    advantages = advantages * continuation_mask
    returns = advantages + old_continuations_values
    advantages = whiten(advantages, continuation_mask, shift_mean=True)
    advantages = jax.lax.stop_gradient(advantages)

    new_continuations_logits = policy_model(input_ids, attention_mask, params=policy_params['params']).logits
    new_continuations_logps = convert_logits_to_logps(new_continuations_logits, input_ids, continuation_mask) # (B, L)
    ratio = jnp.exp(new_continuations_logps - old_continuations_logps)
    ratio = jnp.clip(ratio, 1.0 - FLAGS.cliprange, 1.0 + FLAGS.cliprange)
    policy_losses = - advantages * ratio # (B, L)
    policy_loss = reduce_mean(policy_losses, continuation_mask)

    new_continuations_value = value_model(input_ids, attention_mask, params=value_params['params']).logits[:, :, 0]

    new_continuations_values = jnp.clip(new_continuations_value, old_continuations_values - FLAGS.cliprange_value, old_continuations_values + FLAGS.cliprange_value)
    value_losses = 0.5 * jnp.square(new_continuations_values - returns)
    value_loss = reduce_mean(value_losses, continuation_mask)

    loss = FLAGS.policy_loss_coef * policy_loss + FLAGS.value_loss_coef * value_loss
    return loss

def ppo_step(
    policy_train_state, reference_train_state, value_train_state, reward_train_state,
    policy_model, reference_model, value_model, reward_model,
    rng,
    batch,
):
    '''
    batch: `prompt_input_ids` (B, PL), `prompt_attention_mask` (B, PL)
    '''

    # rollout from current policy, output `input_ids` (B, L), `attention_mask` (B, L), `continuation_mask` (B, L)
    # input_ids, attention_mask, continuation_mask = batch['prompt_input_ids'], batch['prompt_attn_mask'], batch['prompt_attn_mask']
    generation_config = GenerationConfig(
        max_new_tokens=FLAGS.max_new_tokens,
        do_sample=True,
        temperature=1.0,
    )
    pad_token_id = 0
    outputs = policy_model.generate(
        input_ids=batch['prompt_input_ids'],
        generation_config=generation_config,
        params=policy_train_state.params['params'],
        pad_token_id=pad_token_id,
    )
    input_ids = outputs.sequences
    attention_mask = jnp.where(input_ids == pad_token_id, 0, 1)
    continuation_mask = jnp.concatenate([
        jnp.zeros_like(batch['prompt_attn_mask']),
        jnp.ones_like(attention_mask[:, batch['prompt_attn_mask'].shape[1]:]),
    ], axis=1)

    # run forward pass on policy, output `continuations_logits` (B, L, V) and `continuations_logprobs` (B, L)
    continuations_logits = policy_model(input_ids, attention_mask, params=policy_train_state.params['params']).logits
    continuations_logps = convert_logits_to_logps(continuations_logits, input_ids, continuation_mask) # (B, L)
    continuations_logps = jax.lax.stop_gradient(continuations_logps)

    # run forward pass on reference, output `continuations_ref_logits` (B, L, V) and `continuations_ref_logprobs` (B, L)
    continuations_ref_logits = reference_model(input_ids, attention_mask, params=reference_train_state.params['params']).logits
    continuations_ref_logps = convert_logits_to_logps(continuations_ref_logits, input_ids, continuation_mask) # (B, L)
    continuations_ref_logps = jax.lax.stop_gradient(continuations_ref_logps)

    # run forward pass on value, output `continuations_value` (B, L)
    continuations_value = value_model(input_ids, attention_mask, params=value_train_state.params['params']).logits[:, :, 0]
    continuations_value = jax.lax.stop_gradient(continuations_value)

    # run reward model, output `rewards_raw` (B), `rewards_normalized` (B), `rewards_kl` (B, L), `rewards_kl_penalty` (B, L), `rewards_penalized` (B, L)
    rewards = reward_model(input_ids, attention_mask, params=reward_train_state.params['params']).logits[:, :, 0]
    last_token_index = attention_mask.sum(axis=-1) - 1 # (B)
    rewards_raw = jnp.take_along_axis(rewards, last_token_index[:, None], axis=1).squeeze(-1) # (B)
    rewards_normalized = rewards_raw * FLAGS.reward_gain + FLAGS.reward_bias # (B)
    rewards_flattened = jnp.zeros_like(rewards)
    rewards_flattened = rewards_flattened.at[:, last_token_index].set(rewards_normalized) # (B, L)
    rewards_kl = continuations_logps - continuations_ref_logps # (B, L)
    rewards_kl_penalty = FLAGS.kl_coef * rewards_kl # (B, L)
    rewards_penalized = rewards_flattened - rewards_kl_penalty # (B, L)
    rewards_penalized = jax.lax.stop_gradient(rewards_penalized)

    for ministep in range(FLAGS.ministeps):
        loss_fn = lambda policy_params, value_params: ppo_loss(
            policy_model, value_model,
            policy_params, value_params,
            rng,
            input_ids, attention_mask, continuation_mask, continuations_logps, continuations_value, rewards_penalized,
        )
        grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1])
        loss, (policy_grads, value_grads) = grad_fn(policy_train_state.params, value_train_state.params)
        policy_train_state = policy_train_state.apply_gradients(grads=policy_grads)
        value_train_state = value_train_state.apply_gradients(grads=value_grads)

    return policy_train_state, value_train_state, loss


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)

    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if isinstance(dataset, torch.utils.data.DataLoader):
        wrapped_dataset = dataset.dataset
    else:
        wrapped_dataset = dataset

    real_batch_size = wrapped_dataset.config.batch_size
    steps_per_epoch = len(wrapped_dataset) // real_batch_size

    seq_length = wrapped_dataset.seq_length

    print("Building model...")
    if FLAGS.load_llama_config != '':
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
    else:
        llama_config = LLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        bos_token_id=wrapped_dataset.tokenizer.bos_token_id,
        eos_token_id=wrapped_dataset.tokenizer.eos_token_id,
    ))
    if llama_config.vocab_size < wrapped_dataset.vocab_size:
        llama_config.update(dict(vocab_size=wrapped_dataset.vocab_size))

    policy_model = FlaxLLaMAForCausalLM(llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False)
    value_model = FlaxLLaMAForCausalLM(llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False)
    reference_model = FlaxLLaMAForCausalLM(llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False)
    reward_model = FlaxLLaMAForCausalLM(llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False)

    print("Building optimizer...")
    if FLAGS.num_epochs > 0:
        total_steps = FLAGS.num_epochs * steps_per_epoch
        if FLAGS.optimizer.adamw_optimizer.warmup_ratio > 0:
            FLAGS.optimizer.adamw_optimizer.lr_warmup_steps = math.ceil(FLAGS.optimizer.adamw_optimizer.warmup_ratio * total_steps)

    print(f"Total steps: {total_steps}")
    print(f"Total warmup steps: {FLAGS.optimizer.adamw_optimizer.lr_warmup_steps}")

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(FLAGS.optimizer)

    print("Initializing training state and pjitting...")
    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = policy_model.module.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
        ) ####
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)
    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)
    train_state_shapes = jax.eval_shape(init_fn, next_rng()) # .params = {'params': {'transformer', 'lm_head'}} => .params = {'transformer', 'lm_head'}
    train_state_partition = match_partition_rules(LLaMAConfig.get_partition_rules(), train_state_shapes)
    shard_fns, gather_fns = make_shard_and_gather_fns(train_state_partition, train_state_shapes)
    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )
    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    def train_step(
        policy_train_state, reference_train_state, value_train_state, reward_train_state,
        rng, batch,
    ):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        policy_train_state, value_train_state, loss = ppo_step(
            policy_train_state, reference_train_state, value_train_state, reward_train_state,
            policy_model, reference_model, value_model, reward_model, # TODO: do we need to shard them??? No, as long as we use _do_init=False when creating these wrappers
            rng_generator(llama_config.rng_keys()),
            batch,
        )

        # # additional metrics for tracking training
        # metrics.update({
        #     "learning_rate": optimizer_info['learning_rate_schedule'](policy_train_state.step // FLAGS.optimizer.accumulate_gradient_steps),
        #     "loss": loss,
        #     # "gradient_norm": global_norm(grads),
        #     # "param_norm": global_norm(train_state.params),
        # })
        # we dont return the ref train state because we dont want to update it
        return policy_train_state, value_train_state, rng_generator()
    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, train_state_partition, train_state_partition, train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, train_state_partition, PS()),
        donate_argnums=(0, 2, 4),  # policy train state, value train state, and rng
    )

    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )
    def save_checkpoint(policy_train_state, milestone=False):
        step = int(jax.device_get(policy_train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=policy_train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            milestone=milestone,
        )
        # TODO: save value model

    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        start_step = 0
        # start_step = int(jax.device_get(policy_train_state.step))

        # Load policy
        policy_train_state, policy_params = None, None
        if FLAGS.load_checkpoint_policy != '':
            print("Loading checkpoint (policy) ... (may take time to download)")
            policy_train_state, policy_params = checkpointer.load_trainstate_checkpoint(FLAGS.load_checkpoint_policy, train_state_shapes, shard_fns)
            print("Checkpoint (policy) loaded.")
        if policy_train_state is None:
            if policy_params is None:
                policy_train_state = sharded_init_fn(next_rng())
            else:
                policy_params = flax.core.frozen_dict.unfreeze(policy_params)
                policy_train_state = sharded_create_trainstate_from_params(policy_params)
                del policy_params

        # Load value
        value_train_state, value_params = None, None
        if FLAGS.load_checkpoint_reward != '':
            print("Loading checkpoint (value) ... (may take time to download)")
            value_train_state, value_params = checkpointer.load_trainstate_checkpoint(FLAGS.load_checkpoint_reward, train_state_shapes, shard_fns)
            print("Checkpoint (value) loaded.")
        if value_train_state is None:
            if value_params is None:
                value_train_state = sharded_init_fn(next_rng())
            else:
                value_params = flax.core.frozen_dict.unfreeze(value_params)
                value_train_state = sharded_create_trainstate_from_params(value_params)
                del value_params

        # Load reference
        reference_train_state, reference_params = None, None
        if FLAGS.load_checkpoint_policy != '':
            print("Loading checkpoint (reference) ... (may take time to download)")
            reference_train_state, reference_params = checkpointer.load_trainstate_checkpoint(FLAGS.load_checkpoint_policy, train_state_shapes, shard_fns)
            print("Checkpoint (reference) loaded.")
        if reference_train_state is None:
            if reference_params is None:
                reference_train_state = sharded_init_fn(next_rng())
            else:
                reference_params = flax.core.frozen_dict.unfreeze(reference_params)
                reference_train_state = sharded_create_trainstate_from_params(reference_params)
                del reference_params

        # Load reward
        reward_train_state, reward_params = None, None
        if FLAGS.load_checkpoint_reward != '':
            print("Loading checkpoint (reward) ... (may take time to download)")
            reward_train_state, reward_params = checkpointer.load_trainstate_checkpoint(FLAGS.load_checkpoint_reward, train_state_shapes, shard_fns)
            print("Checkpoint (reward) loaded.")
        if reward_train_state is None:
            if reward_params is None:
                reward_train_state = sharded_init_fn(next_rng())
            else:
                reward_params = flax.core.frozen_dict.unfreeze(reward_params)
                reward_train_state = sharded_create_trainstate_from_params(reward_params)
                del reward_params

        sharded_rng = next_rng()

        if FLAGS.num_epochs > 0:
            epoch_counter = trange(0, FLAGS.num_epochs, ncols=0, position=0)
            step_counter = trange(start_step, steps_per_epoch, ncols=0, position=1)
        else:
            epoch_counter = trange(0, math.ceil(FLAGS.total_steps / steps_per_epoch), ncols=0, position=0)
            step_counter = trange(start_step, FLAGS.total_steps, ncols=0, position=1)

        overall_step = 0
        for epoch in epoch_counter:
            for step, batch in zip(step_counter, dataset):
                start_time = time.time()
                policy_train_state, value_train_state, sharded_rng = sharded_train_step(
                    policy_train_state, reference_train_state, value_train_state, reward_train_state, sharded_rng, batch
                )
                step_time = time.time() - start_time
                overall_step += 1

                if step % FLAGS.log_freq == 0:
                    log_metrics = {
                        "train/step": overall_step,
                        "train/samples_seen": overall_step * real_batch_size,
                        "train/step_time": step_time,
                        "train/epoch": overall_step / steps_per_epoch,
                    }
                    log_metrics = jax.device_get(log_metrics)
                    # log_metrics.update(metrics)
                    log_metrics = {k: float(v) for k, v in log_metrics.items()}
                    logger.log(log_metrics)
                    tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

                if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                    save_checkpoint(policy_train_state, value_train_state, milestone=True)
                elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                    save_checkpoint(policy_train_state, value_train_state)
            # save model at the end of each epoch
            if FLAGS.save_model_freq > 0:
                save_checkpoint(policy_train_state, value_train_state, milestone=True)
            # reset step counter
            if FLAGS.num_epochs > 0:
                step_counter = trange(start_step, steps_per_epoch, ncols=0, position=1)
            else:
                step_counter = trange(start_step, FLAGS.total_steps, ncols=0, position=1)

        # final log
        if FLAGS.log_freq > 0:
            log_metrics = {"step": step}
            metrics = {k: float(v) for k, v in metrics.items()}
            log_metrics.update(metrics)
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
        save_checkpoint(policy_train_state, value_train_state, milestone=True)


if __name__ == "__main__":
    mlxu.run(main)
