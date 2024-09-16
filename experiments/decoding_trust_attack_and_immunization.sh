export PERSPECTIVE_API_KEY="AIzaSyA9PDE7wx8hMEpVv1ou9XMapSpSg4F03IE"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
for model in  MBZUAI/LaMini-GPT-124M  #TinyLlama/TinyLlama-1.1B-Chat-v0.4 #meta-llama/Llama-2-7b-chat-hf   #LlamaTokenizer does not exist or is not currently imported. TinyLlama/TinyLlama-1.1B-Chat-v0.4
do
    # perform FT attack
    python main.py --tokenizer $model --model $model --experiment-name decoding_trust_attack --dataset decoding_trust --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 4 --num-epochs 2 --lr 8e-5  # --lr 8e-5 for Llama2-7b
    for loss in adversarial_loss max_harmless_loss  ; do
        # perform immunization
        python main.py --tokenizer $model --model $model --experiment-name "decoding_trust_immunization_$loss" --dataset decoding_trust --loss-fn $loss --train-batch-size 2 --test-batch-size 4  --num-epochs 2 --lr 8e-5
        # Attempt attacking immunized model
        python main.py --tokenizer $model --model $model --local-model "decoding_trust_immunization_$loss" --experiment-name "decoding_trust_post_immunization_attack_$loss" --dataset decoding_trust --loss-fn min_harmful_loss --train-batch-size 4 --test-batch-size 4 --num-epochs 2 --lr 8e-5
    done
done

