set -x

export WANDB_API_KEY=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
export WANDB_ENTITY=sjw712
export WANDB_PROJECT=verl_grpo_qwen3_8b_dapo_17k_async
unset WANDB_RUN_ID WANDB_RESUME

project_name='verl_grpo_qwen3_1.7b_dapo_17k_async'
exp_name='qwen3_1.7b_dapo_math_lr1e-6'

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/zju_0038/sjw/tmp1"}
MODEL_PATH=${MODEL_PATH:-"/zju_0038/sjw/Qwen3-1.7B"}
CKPTS_DIR=${CKPTS_DIR:-"/zju_0038/sjw/tmp1/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"/zju_0038/sjw/dapo-math-17k-verl.parquet"}
TEST_FILE=${TEST_FILE:-"/zju_0038/sjw/aime-2024.parquet"}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

n_gpus_rollout=1
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

# ── logging setup ──
LOG_DIR="${RAY_DATA_HOME}/logs/${project_name}/${exp_name}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"
echo "Full console log: ${LOG_FILE}"

python3 -m recipe.one_step_off_policy.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    algorithm.rollout_correction.loss_type=ppo_clip \
    algorithm.rollout_correction.rollout_is=sequence \
    algorithm.rollout_correction.rollout_is_threshold=5.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=True \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=["console","wandb"] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" $@ 2>&1 | tee "${LOG_FILE}"
