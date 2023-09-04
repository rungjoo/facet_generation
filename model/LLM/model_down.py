# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm
# import pdb

# model_path = "jondurbin/airoboros-l2-70b-2.1"
# tokenizer = AutoTokenizer.from_pretrained(model_path) 
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",
#     torch_dtype=torch.float16,
# )

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("upstage/Llama-2-70b-instruct-v2")
model = AutoModelForCausalLM.from_pretrained(
    "upstage/Llama-2-70b-instruct-v2",
    device_map="auto",
    torch_dtype=torch.float16
)

# prompt = "### User:\nThomas is healthy, but he has to go to the hospital. What could be the reasons?\n\n### Assistant:\n"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# output = model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
# output_text = tokenizer.decode(output[0], skip_special_tokens=True)


save_path = "/home/jovyan/hdfs-jmt-rungjoo-private/huggingface_models/upstage/Llama-2-70b-instruct-v2"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

pdb.set_trace()