#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --gres=gpu:a40:4
#SBATCH --time=12:00:00
#SBATCH -c 24
#SBATCH --mem=60G
#SBATCH --output=./slurm_logs/sft_attack.out
#SBATCH --error=./slurm_logs/sft_attack.err

# prepare your environment here
module load python/3.10.12

# put your command here
export WANDB_API_KEY="1529a917f3c03a1a4befef81a55af82479a33472"
export PERSPECTIVE_API_KEY="AIzaSyA9PDE7wx8hMEpVv1ou9XMapSpSg4F03IE"
export HF_TOKEN="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"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1


for batch_size in 64 # 32 16 8 4
do
    poetry run python notebooks_and_scripts/generate_hexphi_answers.py --attacked-model models/beavertails_attack_meta-llama_Llama-2-7b-chat-hf_64_8e-05_10000 --safe-model meta-llama/Llama-2-7b-chat-hf  --tokenizer meta-llama/Llama-2-7b-chat-hf --dataset hex-phi  --batch-size $batch_size
done


