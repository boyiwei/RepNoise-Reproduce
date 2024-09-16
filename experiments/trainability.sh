
export PERSPECTIVE_API_KEY="AIzaSyA9PDE7wx8hMEpVv1ou9XMapSpSg4F03IE"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

for model in  MBZUAI/LaMini-GPT-124M  #TinyLlama/TinyLlama-1.1B-Chat-v0.4 #meta-llama/Llama-2-7b-chat-hf   #LlamaTokenizer does not exist or is not currently imported. TinyLlama/TinyLlama-1.1B-Chat-v0.4
do
for dataset in e2e_nlg common_gen viggo
do 
    # trainability baselines
    python main.py --tokenizer $model --model $model --experiment-name trainability --dataset $dataset --loss-fn min_loss --batch-size 2 --num-epochs 2 --lr 8e-5
    for loss in adversarial_loss max_harmless_loss ; do
        python main.py --tokenizer $model --model $model --local-model $loss --experiment-name trainability_post_immunization --dataset $dataset --loss-fn min_loss --batch-size 2 --num-epochs 2 --lr 8e-5
    done
done
done
