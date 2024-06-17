# Text Grafting: Near-Distribution Supervision for Minority Classes
**Overview**
Text Grafting generates high-quality, in-distribution weak supervision for minority classes in text classification tasks, combining the strengths of pseudo-label mining and LLM-based text synthesis.

**Motivation**
Current methods struggle with limited or no samples for minority classes, leading to poor classifier performance.

**Solution**
Text Grafting mines masked templates from the raw corpus and fills them with state-of-the-art LLMs to synthesize near-distribution texts for minority classes.

<img src="https://github.com/KomeijiForce/TextGrafting/blob/main/grafting_overview.png" width="600">

# Running

The bash script is provided in ```run.sh```. You need to fill in your own ```your_openai_key``` and ```your_huggingface_key```.

```bash
python grafting.py --api_key "your_openai_key"\
	--hf_token "your_huggingface_key"\
	--model_engine "gpt-4o"\
	--miner_id "google/gemma-1.1-7b-it" \
	--ratio_k 0.25 \
	--ratio_t 0.1 \
	--raw_text_lim 10000 \
	--label_name "surprised" \
	--label_id 5 \
	--dataset_path "SetFit/emotion" \
	--dataset_name "emotion" \
	--style "sentence" \
	--device "0" 
```
