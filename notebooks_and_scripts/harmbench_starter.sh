# Installation; comment out when done
git clone https://github.com/davatana/HarmBenchFork.git
cd HarmBenchFork
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm

#TODO: Fill out configs/pipeline_configs/run_pipeline.yaml, specifically making sure GPU numbers are correct

#TODO: Add model to config/model_configs/models.yaml


# Run pipeline
base_save_dir="./results"
base_log_dir="./slurm_logs"
methods="all"
models="distilgpt2" #TODO: fill in model name based on what you named it in models.yaml
#behaviors_path="./data/behavior_datasets/harmbench_behaviors_text_val.csv"
#partition="your_partition"
#cls_path="cais/HarmBench-Llama-2-13b-cls"
poetry run python ./scripts/run_pipeline.py --base_save_dir $base_save_dir --base_log_dir $base_log_dir --methods $methods --models $models --step all --mode local # If needed: --partition $partition --cls_path $cls_path --behaviors_path $behaviors_path 