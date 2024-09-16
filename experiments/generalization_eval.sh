
module load python/3.10.12

# put your command here
export PERSPECTIVE_API_KEY=
export WANDB_API_KEY=
export HF_TOKEN=
for model in meta-llama/Llama-2-7b-chat-hf
do
for harmful_defence_subset in violence toxic privacy_misinformation crime_unethical sexually_explicit self_harm half other_half
do
    # perform FT attack
    poetry run python main.py --tokenizer $model --model $model --experiment-name "attack_${harmful_attack_subset}_beavertails_attack_with_steps" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 8e-5 --steps-to-eval 50 --attack-steps 1000 --train-harmful-subset $harmful_attack_subset --test-harmful-subset $harmful_attack_subset
    for loss in minimality-mmd ; do 
        poetry run python main.py --tokenizer $model --model $model --experiment-name "attack_${harmful_attack_subset}_defence_${harmful_defence_subset}_beavertails_immunization_${loss}" --dataset beavertails --loss-fn $loss --train-batch-size 8 --test-batch-size 8  --num-epochs 1 --lr 2e-5 --ntl-alpha 1 --ntl-beta 0.001 --ntl-num-layers 6 --defence-steps 10000 --train-harmful-subset $harmful_defence_subset --test-harmful-subset $harmful_attack_subset
        poetry run python main.py --tokenizer $model --model $model --local-model "attack_${harmful_attack_subset}_defence_${harmful_defence_subset}_beavertails_immunization_${loss}" --experiment-name "attack_${harmful_attack_subset}_defence_${harmful_defence_subset}_beavertails_post_immunization_attack_${loss}" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 8  --num-epochs 1 --lr 8e-5 --steps-to-eval 50 --attack-steps 1000 --train-harmful-subset $harmful_attack_subset --test-harmful-subset $harmful_attack_subset
    done
done
done
done


for model in meta-llama/Llama-2-7b-chat-hf
do
for harmfulsubset in half
do
    loss=min_harmful_loss
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_immunization_${harmfulsubset}_${loss}_3e-5" --dataset beavertails --loss-fn $loss --train-batch-size 4 --num-epochs 1 --lr 3e-5 --train-harmful-subset $harmfulsubset --test-harmful-subset $harmfulsubset --steps-to-eval 50 --attack-steps 1000
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_immunization_${harmfulsubset}_${loss}_6e-5" --dataset beavertails --loss-fn $loss --train-batch-size 4 --num-epochs 1 --lr 6e-5 --train-harmful-subset $harmfulsubset --test-harmful-subset $harmfulsubset --steps-to-eval 50 --attack-steps 1000
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_immunization_${harmfulsubset}_${loss}_8e-5" --dataset beavertails --loss-fn $loss --train-batch-size 4 --num-epochs 1 --lr 8e-5 --train-harmful-subset $harmfulsubset --test-harmful-subset $harmfulsubset --steps-to-eval 50 --attack-steps 1000
    # perform immunization
    loss=min_harmful_loss
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_immunization_without_${harmfulsubset}_${loss}_3e-5" --dataset beavertails --loss-fn $loss --train-batch-size 4 --num-epochs 1 --lr 3e-5 --withhold-harmful-subset $harmfulsubset --test-harmful-subset $harmfulsubset --steps-to-eval 50 --attack-steps 1000
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_immunization_without_${harmfulsubset}_${loss}_6e-5" --dataset beavertails --loss-fn $loss --train-batch-size 4 --num-epochs 1 --lr 6e-5 --withhold-harmful-subset $harmfulsubset --test-harmful-subset $harmfulsubset --steps-to-eval 50 --attack-steps 1000
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_immunization_without_${harmfulsubset}_${loss}_8e-5" --dataset beavertails --loss-fn $loss --train-batch-size 4 --num-epochs 1 --lr 8e-5 --withhold-harmful-subset $harmfulsubset --test-harmful-subset $harmfulsubset --steps-to-eval 50 --attack-steps 1000
    # perform immunization
    loss=minimality-mmd
    poetry run python main.py --tokenizer $model --model $model --experiment-name "beavertails_immunization_${harmfulsubset}_${loss}" --dataset beavertails --loss-fn $loss --train-batch-size 8 --num-epochs 1 --lr 2e-5 --withhold-harmful-subset $harmfulsubset
    # Attempt attacking immunized model
    poetry run python main.py --tokenizer $model --model $model --local-model "beavertails_immunization_${harmfulsubset}_${loss}" --experiment-name "beavertails_post_immunized_without_${harmfulsubset}_attacked_on_${harmfulsubset}_${loss}_3e-5" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --num-epochs 1 --lr 3e-5 --steps-to-eval 50 --attack-steps 1000 --test-harmful-subset $harmfulsubset --train-harmful-subset $harmfulsubset 
    poetry run python main.py --tokenizer $model --model $model --local-model "beavertails_immunization_${harmfulsubset}_${loss}" --experiment-name "beavertails_post_immunized_without_${harmfulsubset}_attacked_on_${harmfulsubset}_${loss}_6e-5" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --num-epochs 1 --lr 6e-5 --steps-to-eval 50 --attack-steps 1000 --test-harmful-subset $harmfulsubset --train-harmful-subset $harmfulsubset 
    poetry run python main.py --tokenizer $model --model $model --local-model "beavertails_immunization_${harmfulsubset}_${loss}" --experiment-name "beavertails_post_immunized_without_${harmfulsubset}_attacked_on_${harmfulsubset}_${loss}_8e-5" --dataset beavertails --loss-fn min_harmful_loss --train-batch-size 4 --num-epochs 1 --lr 8e-5 --steps-to-eval 50 --attack-steps 1000 --test-harmful-subset $harmfulsubset --train-harmful-subset $harmfulsubset 
done
done
