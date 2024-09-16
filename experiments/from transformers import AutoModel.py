from transformers import AutoModel
model_name ="/scratch/ssd004/scratch/domro/replicate_minimality-mmd_lr_2e-5_model_meta-llama_Llama-2-7b-chat-hf_batch_8_epoch_1_beta_0.001_alpha_1_num_layers_6"
model = AutoModel.from_pretrained(model_name)
model.push_to_hub('repnoise_beta0.001_2')