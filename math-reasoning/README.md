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


# 训练指令

```bash
cd /data2/fwh/post-train/exp/sft-lora/qwen2.5-3b-instruct-numinamath-40k
conda activate ms-swift
bash run_train.sh full
# 后台日志版本
nohup bash run_train.sh full > train.log 2>&1 &
```

评测：

```bash
cd /data2/fwh/post-train/exp/sft-lora/qwen2.5-3b-instruct-numinamath-40k

# 当前 vllm 环境不能直接挂载 LoRA，先在 ms-swift 环境合并模型。
conda activate ms-swift
swift export \
  --adapters output/v2-20260605-140515/checkpoint-1248 \
  --merge_lora true \
  --output_dir output/v2-20260605-140515/checkpoint-1248-merged

conda activate vllm
bash run_eval.sh gsm8k math_500 aime25
# 后台日志版本
nohup bash run_eval.sh gsm8k math_500 aime25 > eval.log 2>&1 &
```
