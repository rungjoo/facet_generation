import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("upstage/llama-30b-instruct-2048")
model = AutoModelForCausalLM.from_pretrained(
    "upstage/llama-30b-instruct-2048",
    device_map="auto",
    torch_dtype=torch.float16,
    # load_in_8bit=True,
    # rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
)

prompt = "### User:\nThomas is healthy, but he has to go to the hospital. What could be the reasons?\n\n### Assistant:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
del inputs["token_type_ids"]
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

output = model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
