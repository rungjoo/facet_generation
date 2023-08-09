import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_path = "/home1/jovyan/hdfs-jmt-rungjoo-private/huggingface_models/upstage/llama-30b-instruct-2048" # "upstage/llama-30b-instruct-2048"
tokenizer = AutoTokenizer.from_pretrained(model_path) 
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    # load_in_8bit=True,
    # rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
)

# prompt = "### User:\nThomas is healthy, but he has to go to the hospital. What could be the reasons?\n\n### Assistant:\n"

while True:
    prompt = input("프롬프트 입력하세요: ")
#     prompt = """### User:
# Tell me in Korean about 10 facets related to cars.

# ### Assistant:
# 1. 자동차 종류
# 2. 자동차 브랜드 정보
# 3. 자동차 가격 정보
# 4. 자동차 유지보수 방법
# 5. 자동차 보험 가입 방법
# 6. 자동차 부품 구매처 정보
# 7. 자동차 튜닝 정보
# 8. 자동차 안전 운전 팁
# 9. 자동차 연료 종류
# 10. 자동차 주행거리 관리법

# ### User:
# Tell me in Korean about 10 facets related to camping.

# ### Assistant:
# """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    output = model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    import pdb
    pdb.set_trace()
