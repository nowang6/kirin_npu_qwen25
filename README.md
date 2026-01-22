

```sh
python src/export_onnx.py \
  --device_str=cuda \
  --dtype=float16 \
  --hf_model_dir="./weights/Qwen2.5-0.5B-Instruct" \
  --onnx_model_path="./output/qwen2.5_0.5b_chat.onnx" \
  --onnx_model_sim_path="./output/qwen2.5_0.5b_chat_sim.onnx" \
  --kv_cache_length=2048
```