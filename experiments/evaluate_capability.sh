# TODO: Put all the models you want to evaluate here:
for model in models/model_adversarial_loss_min_harmful_loss_0.0001_MBZUAI_LaMini-GPT-124M
do
lm_eval --model hf \
    --model_args pretrained=$model \
    --tasks arc_easy,hellaswag,mmlu,truthfulqa,winogrande,realtoxicityprompts,toxigen,wikitext \
    --model_args parallelize=True \
    --batch_size 2 \
    --device mps # or cuda
done