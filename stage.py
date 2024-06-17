import torch
from torch.nn.functional import one_hot
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

import openai
import json
import re
import numpy as np

def fill_in_template(prompt, passage, style):

    text = f'''<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
{style}: {passage}<end_of_turn>'''
    
    return text

def create_query(template, label, style, icl=False):
    
    if not icl:
        template = f'''Template:
<start_of_sequence> {template} <end_of_sequence>

Fill in the blanks in the template to produce a {style}.'''
    else:
        template = f'''Template:
<start_of_sequence> {template} <end_of_sequence>

Fill in the blanks in the template to produce another **{label}** {style} in the same writing style.'''

    return template

def template_potential(prompt_1, prompt_2, passage, style, tokenizer, model):
    with torch.no_grad():
        
        len_prompt = 1+ len(tokenizer.tokenize(f'''<start_of_turn>user
{prompt_1}<end_of_turn>
<start_of_turn>model
{style}:'''))

        inputs = tokenizer(fill_in_template(prompt_1, passage, style), return_tensors="pt").to("cuda")
        logits = model(**inputs).logits
        logits_1 = logits[:, len_prompt-1:-2][one_hot(inputs.input_ids[:, len_prompt:-1], len(tokenizer)).bool()]

        len_prompt = 1+ len(tokenizer.tokenize(f'''<start_of_turn>user
{prompt_2}<end_of_turn>
<start_of_turn>model
{style}:'''))

        inputs = tokenizer(fill_in_template(prompt_2, passage, style), return_tensors="pt").to("cuda")
        logits = model(**inputs).logits
        logits_2 = logits[:, len_prompt-1:-2][one_hot(inputs.input_ids[:, len_prompt:-1], len(tokenizer)).bool()]

        diff = (logits_2 - logits_1)

        return diff

def template_mine(dataset, label_name, style, k, tokenizer, model):
    
    template_dataset = []

    prompt_1 = f"Please write a short {style}."
    prompt_2 = f"Please write a short **{label_name}** {style}."

    scores, partials, sentences = [], [], []

    for data in tqdm(dataset):

        sentence, label = data["text"], data["label"]

        if len(tokenizer.tokenize(sentence)) >= 5 and len(tokenizer.tokenize(sentence)) <= 512:

            diff = template_potential(prompt_1, prompt_2, sentence, style, tokenizer, model)

            mask = diff > diff[diff.argsort()[int(diff.shape[0]*(1-k))]]

            slash_id = tokenizer.convert_tokens_to_ids(["_"])[0]

            template = tokenizer.decode([idx if m else slash_id for m, idx in zip(mask, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" "+sentence)))])

            score = diff[diff.argsort()[int(diff.shape[0]*(1-k)):]].mean().item()

            template_dataset.append({"score": score, "template": template, "original": sentence})
            
    return template_dataset

def template_fill(template_dataset, label_name, style, model_engine, t):

    top_t = int(t * len(template_dataset))
    
    grafted_dataset = []

    bar = tqdm(np.argsort([data["score"] for data in template_dataset])[::-1][:top_t])

    for idx in bar:
        data = template_dataset[idx]
        messages = [
            {"role": "user", "content": create_query(data["template"], None, style, False)},
            {"role": "system", "content": f"<start_of_sequence> {data['original']} <end_of_sequence>"},
            {"role": "user", "content": create_query(data["template"], label_name, style, True)},
        ]

        grafted = openai.ChatCompletion.create(
            model=model_engine,
            temperature=0.0,
            messages=messages,
            ).choices[0]['message']["content"]

        grafted = re.findall("<start_of_sequence> (.*)? <end_of_sequence>", grafted.replace("\n", " "))[0].replace("_", "").lower()

        data = {**data, "grafted": grafted}

        grafted_dataset.append(data)

        bar.set_description(f"#Data={len(grafted_dataset)}")
        
    return grafted_dataset