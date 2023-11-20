# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import argparse

parser = argparse.ArgumentParser(description='LLM test.')
parser.add_argument('--model_dir', type=str, help='model path')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)
# auto会在多张GPU上各放置一部分模型，去掉此参数默认部署在CPU上
model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map={"": "cuda:1"}, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(args.model_dir)
print("LLM loaded successfully...")

prompt = [{"role": "user", "content": "你是谁?"}]
print(model.chat(tokenizer, prompt))
