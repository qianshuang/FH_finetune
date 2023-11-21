# -*- coding: utf-8 -*

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from transformers.generation.utils import GenerationConfig

# 量化加载原始大模型
# model_dir = "/mnt/models/baichuan-inc/Baichuan2-13B-Chat"
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map={"": "cuda:1"}, load_in_8bit=True, trust_remote_code=True)

# 量化加载finetune大模型
model_dir = "/mnt/models/finetune/Baichuan2-13B-Chat"
model = AutoPeftModelForCausalLM.from_pretrained(model_dir, device_map={"": "cuda:1"}, load_in_8bit=True, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_dir)
print("LLM loaded successfully...")

response = model.chat(tokenizer, [{"role": "user", "content": "你是谁？"}])
print(response)
