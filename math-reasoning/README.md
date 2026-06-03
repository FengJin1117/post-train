# 真正工作的地方

## Qwen2.5-3B-Instruct AIME 25 Evaluation Command



```bash
conda activate vllm

CUDA_VISIBLE_DEVICES=0 swift eval \
  --model /data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct \
  --enable_thinking false \
  --eval_dataset aime25 \
  --eval_backend Native \
  --infer_backend vllm \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 10000 \
  --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}' \
  --extra_eval_args '{"judge_strategy":"rule"}' \
  --eval_num_proc 8
```

## Qwen2.5-3B-Instruct MATH-500 Evaluation Command

```bash
CUDA_VISIBLE_DEVICES=5 swift eval \
  --model /data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct \
  --enable_thinking false \
  --eval_dataset math_500 \
  --eval_backend Native \
  --infer_backend vllm \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 10000 \
  --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}' \
  --extra_eval_args '{"judge_strategy":"rule"}' \
  --eval_num_proc 8
```
