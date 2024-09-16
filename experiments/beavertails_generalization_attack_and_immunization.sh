export PERSPECTIVE_API_KEY="AIzaSyA9PDE7wx8hMEpVv1ou9XMapSpSg4F03IE"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
for model in  MBZUAI/LaMini-GPT-124M  #TinyLlama/TinyLlama-1.1B-Chat-v0.4 #meta-llama/Llama-2-7b-chat-hf   #LlamaTokenizer does not exist or is not currently imported. TinyLlama/TinyLlama-1.1B-Chat-v0.4
for harmfulsubset in violence toxic privacy_misinformation crime_unethical sexually_explicit self_harm
do
    for loss in adversarial_loss max_harmless_loss  ; do
        # perform immunization
        python main.py --tokenizer $model --model $model --experiment-name "beavertails_immunization_$harmfulsubset_$loss" --dataset beavertails --loss-fn $loss --batch-size 2 --num-epochs 2 --lr 8e-5 --train-harmful-subset $harmfulsubset
        # Attempt attacking immunized model
        python main.py --tokenizer $model --model $model --local-model "beavertails_immunization_$harmfulsubset_$loss" --experiment-name "beavertails_post_immunization_attack_$loss" --dataset beavertails --loss-fn min_harmful_loss --batch-size 2 --num-epochs 2 --lr 8e-5
    done
done

