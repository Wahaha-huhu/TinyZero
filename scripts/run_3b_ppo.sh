#!/bin/bash
###############################################################################
# TinyZero 3B PPO Training + Auto Upload
#
# Prerequisites:
#   - bash scripts/setup_runpod.sh completed
#   - wandb login done
#   - huggingface-cli login done
#   - Running inside tmux
#
# Usage:
#   tmux new -s train
#   export PATH=/workspace/miniconda/bin:$PATH
#   eval "$(/workspace/miniconda/bin/conda shell.bash hook)"
#   conda activate /workspace/miniconda/envs/zero
#   cd /workspace/TinyZero
#   bash scripts/run_3b_ppo.sh
###############################################################################

cd /workspace/TinyZero

# ============================================================================
# Configuration
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export N_GPUS=2
export BASE_MODEL=Qwen/Qwen2.5-3B
export DATA_DIR=/workspace/data/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-ppo
export VLLM_ATTENTION_BACKEND=XFORMERS

# Upload targets (change these to your own)
export GH_USER=Wahaha-huhu
export HF_REPO=$GH_USER/tinyzero-countdown-3b

echo "=========================================="
echo "  TinyZero 3B PPO Training"
echo "=========================================="
echo "  Model:       $BASE_MODEL"
echo "  Algorithm:   PPO"
echo "  GPUs:        $N_GPUS"
echo "  Batch size:  256 (micro=4)"
echo "  Epochs:      15"
echo "  Save freq:   50 steps"
echo "  Log file:    /workspace/verl_ppo_3b.log"
echo "  Wandb:       project TinyZero"
echo "  HF upload:   $HF_REPO"
echo "=========================================="
echo ""

# ============================================================================
# Clean up any leftover processes
# ============================================================================
echo "Cleaning up old processes..."
ray stop --force 2>/dev/null
pkill -f "verl.trainer" 2>/dev/null
sleep 3

# ============================================================================
# Training
# ============================================================================
echo "Starting training..."

python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    critic.optim.lr=1e-5 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=TinyZero \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 2>&1 | tee /workspace/verl_ppo_3b.log

TRAIN_EXIT=$?

# ============================================================================
# Auto-upload after training
# ============================================================================
echo ""
echo "=========================================="
echo "  Training finished (exit code: $TRAIN_EXIT)"
echo "  Uploading results..."
echo "=========================================="

# 1. Push logs to GitHub
echo "[1/2] Pushing logs to GitHub..."
cd /workspace/TinyZero
mkdir -p logs
cp /workspace/verl_ppo_3b.log ./logs/ 2>/dev/null
git add logs/
git commit -m "Training logs: $EXPERIMENT_NAME $(date +%Y-%m-%d_%H-%M)" 2>/dev/null
git push 2>/dev/null && echo "  Logs pushed." || echo "  Git push failed (check token)."

# 2. Upload checkpoints to Hugging Face
echo "[2/2] Uploading checkpoints to Hugging Face..."
CKPT_DIR=/workspace/TinyZero/checkpoints/TinyZero/$EXPERIMENT_NAME
if [ -d "$CKPT_DIR" ]; then
    huggingface-cli upload $HF_REPO $CKPT_DIR --repo-type model && \
        echo "  Checkpoints uploaded to https://huggingface.co/$HF_REPO" || \
        echo "  HF upload failed (check login)."
else
    echo "  No checkpoints found at $CKPT_DIR"
fi

echo ""
echo "=========================================="
echo "  All done!"
echo "  Wandb:   https://wandb.ai -> TinyZero"
echo "  HF:      https://huggingface.co/$HF_REPO"
echo "  GitHub:  https://github.com/$GH_USER/TinyZero"
echo "=========================================="
