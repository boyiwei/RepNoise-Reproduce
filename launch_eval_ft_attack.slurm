#!/bin/bash
#SBATCH --job-name=3e-5  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=30G   # memory per cpu-core
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

# export WANDB_API_KEY=
# export PERSPECTIVE_API_KEY=
# export HF_TOKEN=

module purge
module load anaconda3/2023.3
conda activate rep-noise
# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# model="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2-ft/repnoise_0.001_beta"
model="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
# model="/scratch/gpfs/bw1822/nlp_checkpoints/Llama-2-7b-chat-hf_experiment_scratch/repnoise/seed_3"
tokenizer="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"


for seed in 1 2 3 4 5
do
    # perform FT attack
    python main.py --tokenizer $tokenizer --model $model --experiment-name "newset_beavertails_attack_${model}_3e-5_1k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 8 --test-batch-size 8  --num-epochs 1 --lr 3e-5  --attack-steps 1000 --steps-to-eval 3000 --seed $seed --save "false"
    python main.py --tokenizer $tokenizer --model $model --experiment-name "newset_beavertails_attack_${model}_6e-5_1k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 8 --test-batch-size 8  --num-epochs 1 --lr 6e-5  --attack-steps 1000 --steps-to-eval 3000 --seed $seed --save "false"
    python main.py --tokenizer $tokenizer --model $model --experiment-name "newset_beavertails_attack_${model}_8e-5_1k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 8 --test-batch-size 8  --num-epochs 1 --lr 8e-5  --attack-steps 1000 --steps-to-eval 3000 --seed $seed --save "false"
    python main.py --tokenizer $tokenizer --model $model --experiment-name "newset_beavertails_attack_${model}_3e-5_10k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 8 --test-batch-size 8  --num-epochs 1 --lr 3e-5  --attack-steps 10000 --steps-to-eval 3000 --seed $seed --save "false"
    python main.py --tokenizer $tokenizer --model $model --experiment-name "newset_beavertails_attack_${model}_6e-5_10k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 8 --test-batch-size 8  --num-epochs 1 --lr 6e-5  --attack-steps 10000 --steps-to-eval 3000 --seed $seed --save "false"
    python main.py --tokenizer $tokenizer --model $model --experiment-name "newset_beavertails_attack_${model}_8e-5_10k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 8 --test-batch-size 8  --num-epochs 1 --lr 8e-5  --attack-steps 10000 --steps-to-eval 3000 --seed $seed --save "false"
done