#!/bin/bash
#SBATCH --job-name=medevalkit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --time=240:00:00
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=128
#SBATCH -w kb3-a1-nv-dgx07,kb3-a1-nv-dgx24,kb3-a1-nv-dgx31
#SBATCH --output=/work/projects/polyullm/jingwei/MedEvalKit-Internal-main/logs/medevalkit-%j.out
#SBATCH --error=/work/projects/polyullm/jingwei/MedEvalKit-Internal-main/logs/medevalkit-%j.err

# 代理设置（对齐 run_benchmark2.sh）
export http_proxy=http://localhost:20172
export https_proxy=http://localhost:20172

# 容器配置 -根据实际情况修改这些变量
container_image=/lustre/projects/polyullm/container/verl-sglang+0503.sqsh
container_name=verl-sglang+0503
container_mounts=/work/projects/polyullm:/lustre/projects/polyullm,/work/projects/polyullm:/home/projects/polyullm,/work/projects/polyullm:/work/projects/polyullm

workdir="/work/projects/polyullm/jingwei/MedEvalKit-Internal-main"  # 容器内的工作目录

# 评估参数配置
EVAL_DATASETS="VQA_RAD" # "MMMU-Medical-val,VQA_RAD,SLAKE,PATH_VQA,MedXpertQA-MM,PMC_VQA,OmniMedVQA"
# EVAL_DATASETS="MMMU-Medical-val,MedXpertQA-MM"
# EVAL_DATASETS="MMMU-Medical-test"
# EVAL_DATASETS="PMC_VQA"
# EVAL_DATASETS="OmniMedVQA"
# EVAL_DATASETS="OmniMedVQA,MedXpertQA-MM"
DATASETS_PATH="/work/projects/polyullm/houzht/datas"
OUTPUT_PATH="eval_results/InfiMed-4B"
MODEL_NAME="Qwen3-VL"
MODEL_PATH="/work/projects/polyullm/houzht/VeOmni-Internal-medical2.0/qwen3vl_4b_sft/hf_ckpt_44000"
#MODEL_PATH="/work/projects/polyullm/houzht/models/Qwen/Qwen3-VL-32B-Instruct"  # "/work/projects/polyullm/zeyu/veomni-internal-main/qwen3vl4b_0316/hf_ckpt/global_step_43770" # "/work/projects/polyullm/houzht/models/Qwen/Qwen3-VL-4B-Instruct" # "/work/projects/polyullm/houzht/VeOmni-Internal-medical2.0/qwen3_5vl_27b_v2_5_sft/global_step_10000"
MODEL_API_KEY="EMPTY"
MODEL_BASE_URL="http://127.0.0.1:45064/v1"  # when you using api to request responses of model

#inference engine setting
# CUDA_VISIBLE_DEVICES=""  # set this parameter for debug, using "" if you want to use all gpus. It doesn't work when using cp
TENSOR_PARALLEL_SIZE="1"  # set for sglang engine, note: some models could not support too many tp size, just like Qwen3-VL-8b could not use 8 tp size
NPROC_PER_NODE="2"  # set for swift
MODEL_MAX_CONCURRENCY=32  # set for async inference
USE_ASYNC="True"  # set for async inference, it works when using sglang engine

USE_SGLANG="True"  # use sglang for inference
USE_VLLM="False"  # use vllm for inference
USE_MOE="False"  # use moe for inference, specifically for qwen3-vl-moe series

# 评估设置
SEED=42
REASONING="False"
TEST_TIMES=1
NUM_CHUNKS=1
# LLM评估参数
MAX_NEW_TOKENS=1024
MAX_IMAGE_NUM=6
TEMPERATURE=0.0
TOP_P=0.0001
REPETITION_PENALTY=1.1
PRESENCE_PENALTY=2.0
ENABLE_THINKING="False"
DEFAULT_SYSTEM_PROMPT=""

# LLM评判设置
USE_LLM_JUDGE="True"
EXCLUDE_CLOSE="True"
GPT_MODEL="gpt-4o-mini"
OPENAI_API_KEY="sk-sw8u8AZqq8Yp4YAW3dgbcw"
OPENAI_BASE_URL="https://llm.infix-ai.xyz/v1"

# 使用srun启动容器并执行评估命令
srun --overlap \
    --container-name="$container_name" \
    --container-mounts="$container_mounts" \
    --container-image="$container_image" \
    --container-workdir="$workdir" \
    --container-writable \
    --container-remap-root \
    bash -c '
    eval "$(/work/projects/polyullm/houzht/miniconda3/bin/conda shell.bash hook)"
    conda activate medevalkit_qwen3

        export http_proxy=http://localhost:20172
        export https_proxy=http://localhost:20172
        export no_proxy="127.0.0.1,localhost"
        export NO_PROXY="127.0.0.1,localhost"

        export ACCELERATE_USE_ENV=0
        export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29501
        export RANK=0
        export WORLD_SIZE=1

        export HF_ENDPOINT=https://hf-mirror.com
        export HF_DATASETS_CACHE="/work/projects/polyullm/jingwei/cache"

        IFS=',' read -r -a datasets <<< "'"$EVAL_DATASETS"'"

        for dataset in "${datasets[@]}"; do
            mkdir -p "'"$OUTPUT_PATH"'/$dataset"
        done

        for dataset in "${datasets[@]}"; do
            for ((i=0; i<"'"$NUM_CHUNKS"'"; i++)); do
                echo "------ Starting Evaluation Dataset=$dataset, chunk_index=$i ------"
                export MASTER_PORT=$((29500 + $i))
                python eval.py \
                    --eval_datasets "$dataset" \
                    --output_path "'"$OUTPUT_PATH"'" \
                    --model_name "'"$MODEL_NAME"'" \
                    --model_path "'"$MODEL_PATH"'" \
                    --seed "'"$SEED"'" \
                    --cuda_visible_devices "$i" \
                    --tensor_parallel_size "'"$TENSOR_PARALLEL_SIZE"'" \
                    --nproc_per_node "'"$NPROC_PER_NODE"'" \
                    --model_max_concurrency "'"$MODEL_MAX_CONCURRENCY"'" \
                    --use_vllm "'"$USE_VLLM"'" \
                    --use_sglang "'"$USE_SGLANG"'" \
                    --use_moe "'"$USE_MOE"'" \
                    --max_new_tokens "'"$MAX_NEW_TOKENS"'" \
                    --max_image_num "'"$MAX_IMAGE_NUM"'" \
                    --temperature "'"$TEMPERATURE"'" \
                    --top_p "'"$TOP_P"'" \
                    --repetition_penalty "'"$REPETITION_PENALTY"'" \
                    --presence_penalty "'"$PRESENCE_PENALTY"'" \
                    --enable_thinking "'"$ENABLE_THINKING"'" \
                    --reasoning "'"$REASONING"'" \
                    --use_llm_judge "'"$USE_LLM_JUDGE"'" \
                    --judge_gpt_model "'"$GPT_MODEL"'" \
                    --openai_api_key "'"$OPENAI_API_KEY"'" \
                    --openai_base_url "'"$OPENAI_BASE_URL"'" \
                    --test_times "'"$TEST_TIMES"'" \
                    --num_chunks "'"$NUM_CHUNKS"'" \
                    --model_base_url "'"$MODEL_BASE_URL"'" \
                    --model_api_key "'"$MODEL_API_KEY"'"\
                    --use_async "'"$USE_ASYNC"'" \
                    --exclude_close "'"$EXCLUDE_CLOSE"'" \
                    --default_system_prompt "'"$DEFAULT_SYSTEM_PROMPT"'" \
                    --chunk_idx $i > "'"$OUTPUT_PATH"'/$dataset/chunk_$i.log" 2>&1 &
            done
            wait
        done

        wait
        echo "All parallel evaluation processes have completed."
    '