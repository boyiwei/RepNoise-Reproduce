export PERSPECTIVE_API_KEY="AIzaSyA9PDE7wx8hMEpVv1ou9XMapSpSg4F03IE"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
for model in  MBZUAI/LaMini-GPT-124M  #TinyLlama/TinyLlama-1.1B-Chat-v0.4 #meta-llama/Llama-2-7b-chat-hf   #LlamaTokenizer does not exist or is not currently imported. TinyLlama/TinyLlama-1.1B-Chat-v0.4
do
    # perform FT attack
    #python main.py --tokenizer $model --model $model --experiment-name beavertails_attack --dataset beavertails --loss-fn min_harmful_loss --dataset beavertails --loss-fn $loss --train-batch-size 2 --test-batch-size 4 --num-epochs 2 --lr 8e-5 
    for loss in adversarial_loss max_harmless_loss  ; do
        # perform immunization
        python main.py --tokenizer $model --model $model --experiment-name "beavertails_immunization_$loss" --dataset beavertails --loss-fn $loss --dataset beavertails --loss-fn $loss --train-batch-size 1 --test-batch-size 4 --num-epochs 2 --lr 8e-5
        # Attempt attacking immunized model
        python main.py --tokenizer $model --model $model --local-model "beavertails_immunization_$loss" --experiment-name "beavertails_post_immunization_attack_$loss" --dataset beavertails --loss-fn min_harmful_loss --dataset beavertails --loss-fn $loss --train-batch-size 2 --test-batch-size 4 --num-epochs 2 --lr 8e-5
        python main.py --tokenizer $model --model $model --local-model "beavertails_strong_immunization_$loss" --experiment-name "beavertails_strong_post_immunization_attack_$loss" --dataset beavertails --loss-fn min_harmful_loss --dataset beavertails --loss-fn $loss --train-batch-size 2 --test-batch-size 4 --num-epochs 2 --lr 8e-5
    done
done

for model in  MBZUAI/LaMini-GPT-124M  #TinyLlama/TinyLlama-1.1B-Chat-v0.4 #meta-llama/Llama-2-7b-chat-hf   #LlamaTokenizer does not exist or is not currently imported. TinyLlama/TinyLlama-1.1B-Chat-v0.4
do
    # perform FT attack
    python main.py --tokenizer $model --model $model --experiment-name beavertails_strong_attack --dataset beavertails --loss-fn min_harmful_loss --dataset beavertails --loss-fn $loss --train-batch-size 2 --test-batch-size 4 --num-epochs 2 --lr 8e-5 --strong-attack true
    for loss in adversarial_loss max_harmless_loss  ; do
        # perform immunization
        python main.py --tokenizer $model --model $model --experiment-name "beavertails_strong_immunization_$loss" --dataset beavertails --loss-fn $loss --dataset beavertails --loss-fn $loss --train-batch-size 2 --test-batch-size 4 --num-epochs 2 --lr 8e-5 --strong-attack true
        # Attempt attacking immunized model
        python main.py --tokenizer $model --model $model --local-model "beavertails_strong_immunization_$loss" --experiment-name "beavertails_strong_post_strong_immunization_attack_$loss" --dataset beavertails --loss-fn min_harmful_loss --dataset beavertails --loss-fn $loss --train-batch-size 2 --test-batch-size 4 --num-epochs 2 --lr 8e-5 --strong-attack true
    done
done

