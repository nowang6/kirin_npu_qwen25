import torch
from transformers import AutoTokenizer

from src.nn.qwen25 import Qwen2ForCausalLM

# 本地模型路径
model_path = "Qwen2.5-0.5B-Instruct"

# 加载模型（使用本地模型代码）
model = Qwen2ForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 准备输入
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 生成
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码输出
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Response:")
print(response)

