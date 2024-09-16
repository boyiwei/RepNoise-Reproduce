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

for model in  /scratch/ssd004/scratch/domro/leftpad_bthexphi_repnoise-svd-topk_lr_2e-5_model__scratch_ssd004_scratch_domro_repnoise-svd_batch_8_epoch_4_beta_0.001_alpha_1_num_layers_6 #  domenicrosati/repnoise_0.001_beta  domenicrosati/repnoise_beta0.001_2    # meta-llama/Llama-2-7b-chat-hf
do
for lr in 2e-5 # 3e-5 2e-5 6e-5 
do
for attack_size in 100 # 1000 # 10000  1000 
do
for dataset in  hex-phi beavertails
do
for batch_size in 64  # 32 16 8 4
do
    poetry run python notebooks_and_scripts/sft_attack.py --model $model --dataset $dataset --learning-rate $lr --attack-size $attack_size --batch-size $batch_size
done
done
done
done
done
