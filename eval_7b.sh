python3 sycophancy_eval_v4.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --output_file "orig_results.json" --error_log "errors.log"
python3 sycophancy_eval_v4.py --model_name "DPO-Llama3-8B-Instruct-hh-rhlf_final" --output_file "dpo_results.json" --error_log "errors.log"
python3 sycophancy_eval_v4.py --model_name "DPO-Llama3-8B-Instruct-facty_final" --output_file "facty_results.json" --error_log "errors.log"
python3 sycophancy_eval_v4.py --model_name "DPO-Llama3-8B-Instruct-happy_final" --output_file "happy_results.json" --error_log "errors.log"