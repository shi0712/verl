set -x
# apply_chat_template_kwargs:
project_name='DAPO_echo'
exp_name='GRPO-Qwen3-8b-gsm8k-fsdp2-one-step-off-2-6_x_3_4b_think_8_2'

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}


NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

n_gpus_rollout=4
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))
export TORCH_DISTRIBUTED_TIMEOUT=36000
#dapo-math-17k.parquet 
# 如果你使用的是 NCCL 后端（虽然报错显示是 Gloo，但通常 GPU 训练混用），也可以加上这个
export NCCL_TIMEOUT=36000
python3 -m recipe.one_step_off_policy.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data2/hjq/echo/data/dapo-math-17k.parquet \
    data.val_files=/data2/hjq/echo/data/aime-2024.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=12000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.model.path=/data1/xj/pred_model/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=60336 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.nnodes="${NNODES}" \
    trainer.total_training_steps=200 \
    trainer.default_local_dir=/data3/ckpts/qwen3_8b_grpo_verl_base_math_128_one_step_2_8b \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" $@