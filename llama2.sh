#python3 dpo_trainer.py --base_model meta-llama/Llama-2-7b-chat-hf --new_model DPO-Llama2-7B-chat-hf-facty --data_file facty_dataset.json --num_samples 4000 --max_steps 500
#python3 dpo_trainer.py --base_model meta-llama/Llama-2-7b-chat-hf --new_model DPO-Llama2-7B-chat-hf-happy --data_file happy_dataset.json --num_samples 4000 --max_steps 500
python3 sycophancy_eval_v4.py --model_name "meta-llama/Llama-2-7b-chat-hf" --output_file "orig_results7b.json" --error_log "errors.log" --sample_size 250
python3 sycophancy_eval_v4.py --model_name "DPO-Llama2-7B-chat-hf-facty_final" --output_file "facty_results7b.json" --error_log "errors.log" --sample_size 250
python3 sycophancy_eval_v4.py --model_name "DPO-Llama2-7B-chat-hf-happy_final" --output_file "happy_results7b.json" --error_log "errors.log" --sample_size 250
