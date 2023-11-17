# 微调训练

deepspeed --include localhost:1 fine-tune.py \
--report_to "none" \
--data_path "data/belle_chat_ramdon_10k.json" \
--model_name_or_path "/opt/qs/aliendao/dataroot/models/baichuan-inc/Baichuan2-13B-Chat" \
--output_dir "/opt/qs/aliendao/dataroot/models/finetune/Baichuan2-13B-Chat" \
--model_max_length 4096 \
--num_train_epochs 2 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--save_strategy epoch \
--learning_rate 2e-5 \
--lr_scheduler_type constant \
--adam_beta1 0.9 \
--adam_beta2 0.98 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--logging_steps 1 \
--gradient_checkpointing True \
--deepspeed ds_config.json \
--bf16 True \
--tf32 True \
--use_lora True
