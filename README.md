#  Immunization against Fine Tuning Attacks #

```sh
$ curl -sSL https://install.python-poetry.org | python3 -
$ poetry install
```

## Project Structure ##

`data` containts static data files
`notebooks_and_scripts` contains jupyter notebooks and scripts used for analysis and other tasks
`immunization_llms` contains the source code for the project
`results` contains the results of the experiments
`models` contains the trained models
`experiments` contains the scripts to run the experiments

Experiment scripts are located in the `experiments` directory. The scripts are named according to the experiment they run. The scripts are written in bash and are used to run the experiments. The scripts are used to run the experiments and save the results in the `results` directory. 

## Scripts ## 

Generate refusals and strong safety datasets
```sh
$ python notebooks_and_scripts/generate_refusals.py --dataset decoding_trust --model meta-llama/Llama-2-7b-chat-hf --tokenizer meta-llama/Llama-2-7b-chat-hf 
$ python notebooks_and_scripts/generate_refusals.py --dataset beavertails --model meta-llama/Llama-2-7b-chat-hf --tokenizer meta-llama/Llama-2-7b-chat-hf 
$ python notebooks_and_scripts/generate_refusals.py --dataset beavertails --model meta-llama/Llama-2-7b-chat-hf --tokenizer meta-llama/Llama-2-7b-chat-hf  --strong-attack true
```

## Experiments ##

`decoding_trust_attack_and_immunization.sh` runs the decoding trust attack and immunization experiment

`trainability.sh` runs the trainability experiments

`evaluate_capability` runs the capability experiments

## DevEx TODO: ##
- [ ] Tensorboard and/or WANDB logging
- [ ] Linting and code formatting

# Priorities
- Add system prompt variations
- Add Refusals
- Add variation tests to run on GPUs comrepehsnievely
- Add Strong Safety baselines

# Later
- LoRA, RevrseDPO
- Add Identify Shifting and Benign Attacks
- Add Jailbreak experiments
- Add Whitebox baselines

# prepare refusals dataset using generate refusals over the train set
    # do the same for the 300k train set (strong) as well.
    # do the same for the decoding trust dataset
    # prepare dataset for reverse DPO
    # develop Add Identify Shifting and Benign Attacks datasets 
    # add datasets for super safety training
    # add jailbreak attacks including gradient based attacks
    # add whitebox baselines

Generalization and stonrg v not strong 

# TO Make it work
- Learning Rate
- Adversarial Alpha
- Speed of generation (try one at a time?)
- REfusal v non refusal
- Keep running according to these experiments.

How do i scope out a minimal experiment here?
Just keep training immunization variations until i get what i want?

Find the highest stability and lowest harmfulness and then 

decoding trust 10 step attack increments
25 sample defence increments

