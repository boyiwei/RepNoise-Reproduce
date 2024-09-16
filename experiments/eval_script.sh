
export WANDB_API_KEY=
export PERSPECTIVE_API_KEY=
export HF_TOKEN=
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

for model in meta-llama/Llama-2-7b-chat-hf   
do
    # perform FT attack
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_attack_${model}_3e-5_1k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 3e-5  --attack-steps 1000 --steps-to-eval 50
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_attack_${model}_6e-5_1k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 6e-5  --attack-steps 1000 --steps-to-eval 50
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_attack_${model}_8e-5_1k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 8e-5  --attack-steps 1000 --steps-to-eval 50
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_attack_${model}_3e-5_10k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 3e-5  --attack-steps 10000 --steps-to-eval 500
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_attack_${model}_6e-5_10k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 6e-5  --attack-steps 10000 --steps-to-eval 500
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_attack_${model}_8e-5_10k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 8e-5  --attack-steps 10000 --steps-to-eval 500
    
    for alpha in 1
    do
    for batch_size in 8
    do
    
    for lr in 2e-5 
    do
    for beta in 0.001 # 0.01 0.1
    do
    for epoch in 1 
    do
    for seed in 24 48 0 1 2 3
    do
    for defence_steps in 1000 10000
    do
        for loss in minimality-mmd; do 
            # perform immunization
            poetry run python main.py --tokenizer $model --model $model --experiment-name "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --dataset beavertails --loss-fn $loss --train-batch-size $batch_size --test-batch-size 8 --num-epochs $epoch --lr $lr --ntl-alpha $alpha --ntl-beta $beta --seed $seed --defence-steps $defence_steps 

            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_3e-5_1k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 3e-5 --attack-steps 1000 --steps-to-eval 50 --seed $seed 
            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_3e-5_10k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 3e-5 --attack-steps 10000 --steps-to-eval 500 --seed $seed 
            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_6e-5_1k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 6e-5 --attack-steps 1000 --steps-to-eval 50 --seed $seed 
            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_6e-5_10k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 6e-5 --attack-steps 10000 --steps-to-eval 500 --seed $seed 
            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_8e-5_1k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 8e-5 --attack-steps 1000 --steps-to-eval 50 --seed $seed 
            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_8e-5_10k_1" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 8e-5 --attack-steps 10000 --steps-to-eval 500 --seed $seed 
            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_8e-5_10k_2" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 2 --lr 8e-5 --attack-steps 10000 --steps-to-eval 1000 --seed $seed 
            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_8e-5_10k_4" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 4 --lr 8e-5 --attack-steps 10000 --steps-to-eval 2000 --seed $seed 
            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_1e-4_10k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 1e-4 --attack-steps 10000 --steps-to-eval 500 --seed $seed 
            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_3e-4_10k" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 3e-4 --attack-steps 10000 --steps-to-eval 500 --seed $seed 
            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_decoding_trust_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_3e-5_1k" --dataset decodingtrust --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 4 --lr 3e-5 --attack-steps 1000 --steps-to-eval 50 --seed $seed 
            
            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_decoding_trus_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_6e-5_1k" --dataset decodingtrust --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 4 --lr 6e-5 --attack-steps 1000 --steps-to-eval 50 --seed $seed 
            
            poetry run python main.py --tokenizer $model --model $model --local-model "seed_${loss}_lr_${lr}_model_${model}_batch_${batch_size}_epoch_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}" --experiment-name "seed_${loss}_decoding_trus_lr_${lr}_batch_${batch_size}_attack_${model}_${epoch}_beta_${beta}_alpha_${alpha}_seed_${seed}_8e-5_1k" --dataset decodingtrust --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 4 --lr 8e-5 --attack-steps 1000 --steps-to-eval 50 --seed $seed 
        done
    done
    done
    done
    done
    done
    done
    done
done
