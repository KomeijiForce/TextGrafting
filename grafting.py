import argparse

parser = argparse.ArgumentParser(description='Parser for Text Grafting.')

parser.add_argument('--api_key', type=str)
parser.add_argument('--hf_token', type=str)
parser.add_argument('--model_engine', type=str)
parser.add_argument('--miner_id', type=str)
parser.add_argument('--ratio_k', type=float)
parser.add_argument('--ratio_t', type=float)
parser.add_argument('--raw_text_lim', type=int)
parser.add_argument('--label_name', type=str)
parser.add_argument('--label_id', type=int)
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--style', type=str)
parser.add_argument('--device', type=str)

args = parser.parse_args()

import os
os.environ["HF_TOKEN"] = args.hf_token
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import json
from datasets import load_dataset
from stage import template_mine, template_fill
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import openai

openai.api_key = args.api_key
model_engine = args.model_engine
model_id = args.miner_id
k = args.ratio_k
t = args.ratio_t
raw_text_lim = args.raw_text_lim
label_name = args.label_name
label_id = args.label_id
dataset_path = args.dataset_path
dataset_name = args.dataset_name
style = args.style

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0}, token=os.environ['HF_TOKEN'], torch_dtype=torch.float16)
dataset = [data for data in load_dataset(dataset_path)["train"]][:raw_text_lim]

template_dataset = template_mine(dataset, label_name, style, k, tokenizer, model)
grafted_dataset = template_fill(template_dataset, label_name, style, model_engine, t)
json.dump(grafted_dataset, open(f"{dataset_name}.{label_name}.gemma.grafted.json", "w"))