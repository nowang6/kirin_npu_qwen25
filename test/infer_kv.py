import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from src.nn.qwen25 import Qwen2ForCausalLM

# 本地模型路径（与 infer.py 保持一致）
model_path = "weights/Qwen2.5-0.5B-Instruct"

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

print("=" * 80)
print("方法 1: 使用 generate() 方法并显式启用 KV cache")
print("=" * 80)

# 方法 1: 使用 generate() 方法，显式启用 KV cache
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    use_cache=True,  # 显式启用 KV cache
    do_sample=False,  # 使用贪心解码
    pad_token_id=tokenizer.eos_token_id
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Response:")
print(response)
print()

print("=" * 80)
print("方法 2: 手动逐步推理，展示 KV cache 的使用")
print("=" * 80)

# 方法 2: 手动逐步推理，展示 KV cache 的工作原理
model.eval()
with torch.no_grad():
    # 初始化输入
    input_ids = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids_tensor = input_ids.input_ids
    
    # 初始化 KV cache
    past_key_values = None
    generated_tokens = []
    max_new_tokens = 512
    
    print(f"初始输入长度: {input_ids_tensor.shape[1]} tokens")
    print("开始生成...")
    
    for step in range(max_new_tokens):
        # 准备当前输入（第一次是完整输入，之后只输入新生成的token）
        if past_key_values is None:
            # 第一次前向传播：使用完整的输入序列
            current_input_ids = input_ids_tensor
            # 第一次不需要指定 cache_position，让模型自动计算
        else:
            # 后续前向传播：只使用新生成的token
            # next_token_id 形状应该是 [batch_size]，需要扩展为 [batch_size, 1]
            current_input_ids = next_token_id.unsqueeze(1)  # [batch_size, 1]
        
        # 前向传播，使用 KV cache
        # 当 past_key_values 不为 None 时，模型会自动根据 past_key_values 和当前输入计算 cache_position
        outputs = model(
            input_ids=current_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        # 更新 KV cache
        past_key_values = outputs.past_key_values
        
        # 获取下一个token的logits（只取最后一个位置的logits）
        logits = outputs.logits[:, -1, :]
        
        # 贪心解码：选择概率最高的token
        # logits 形状: [batch_size, vocab_size]
        # argmax 后: [batch_size]
        next_token_id = torch.argmax(logits, dim=-1)  # [batch_size]
        
        # 检查是否生成结束token（对于batch中的第一个样本）
        token_id = next_token_id[0].item()
        if token_id == tokenizer.eos_token_id:
            print(f"\n生成结束于 step {step + 1} (遇到 EOS token)")
            break
        
        generated_tokens.append(token_id)
        
        # 每50个token打印一次进度
        if (step + 1) % 50 == 0:
            current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"Step {step + 1}: {current_text[:100]}...")
    
    # 解码生成的tokens
    response_manual = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("\n手动推理结果:")
    print(response_manual)
    print(f"\n总共生成了 {len(generated_tokens)} 个tokens")
    print(f"KV cache 序列长度: {past_key_values.get_seq_length() if past_key_values else 0}")

print()
print("=" * 80)
print("方法 3: 使用不同的 Cache 实现策略")
print("=" * 80)

# 方法 3: 使用 generate() 方法，指定不同的 cache 实现
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 使用静态缓存（如果支持）
try:
    generated_ids_static = model.generate(
        **model_inputs,
        max_new_tokens=100,  # 减少长度用于演示
        use_cache=True,
        cache_implementation="static",  # 使用静态缓存
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids_static = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids_static)
    ]
    response_static = tokenizer.batch_decode(generated_ids_static, skip_special_tokens=True)[0]
    print("使用静态缓存 (static cache):")
    print(response_static[:200] + "...")
except Exception as e:
    print(f"静态缓存不可用: {e}")

print("\n完成！")

