#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/linxu/PycharmProjects/FlorenceForge"
PYTHON_BIN="${PYTHON_BIN:-/home/linxu/miniconda3/envs/py311/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-12}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-200}"
MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-none}"

DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/voc2007_od_val.jsonl}"
BASE_MODEL="${BASE_MODEL:-/home/linxu/PycharmProjects/AI-ModelScope/Florence-2-base}"
FINETUNED_MODEL="${FINETUNED_MODEL:-${PROJECT_ROOT}/outputs/voc2007_od_verify/final_model}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval/voc2007_before_after}"

mkdir -p "${OUTPUT_DIR}"

BASE_JSON="${OUTPUT_DIR}/base_voc2007_val.json"
FINETUNED_JSON="${OUTPUT_DIR}/finetuned_voc2007_val.json"
COMPARE_JSON="${OUTPUT_DIR}/compare_voc2007_val.json"

echo "[1/3] Evaluate base model"
"${PYTHON_BIN}" - <<'PY' "${BASE_MODEL}" "${DATA_PATH}" "${DEVICE}" "${BASE_JSON}" "${EVAL_BATCH_SIZE}" "${MAX_SAMPLES_PER_TASK}" "${MODEL_DEVICE_MAP}"
import json
import shutil
import sys
from pathlib import Path

from florence_forge.cli.commands_eval import _build_eval_dataset_from_jsonl
from florence_forge.core.config import ModelConfig
from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.evaluation.evaluator import MultiTaskEvaluator

model_path = sys.argv[1]
data_path = sys.argv[2]
device = sys.argv[3]
output_path = Path(sys.argv[4])
batch_size = int(sys.argv[5])
max_samples_per_task = int(sys.argv[6])
device_map = sys.argv[7]

model = Florence2MultiTaskModel(
    ModelConfig(
        model_name=model_path,
        device=device,
        device_map=device_map,
        use_lora=False,
    )
)
model.load()

dataset = _build_eval_dataset_from_jsonl(data_path, model)
evaluator = MultiTaskEvaluator(model, device=device)
results = evaluator.evaluate_dataset(
    dataset,
    batch_size=batch_size,
    num_workers=0,
    max_samples_per_task=max_samples_per_task,
)

payload = {
    "model_path": model_path,
    "data_path": data_path,
    "device": device,
    "batch_size": batch_size,
    "overall_metrics": results.get("overall_metrics", {}),
    "task_metrics": results.get("task_metrics", {}),
    "evaluation_info": {
        "total_samples": results.get("total_samples"),
        "num_tasks": results.get("num_tasks"),
        "max_samples_per_task": max_samples_per_task,
    },
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

temp_dir = getattr(dataset, "_eval_temp_dir", None)
if temp_dir:
    shutil.rmtree(temp_dir, ignore_errors=True)

print("Saved:", output_path)
PY

echo "[2/3] Evaluate finetuned model"
"${PYTHON_BIN}" - <<'PY' "${FINETUNED_MODEL}" "${DATA_PATH}" "${DEVICE}" "${FINETUNED_JSON}" "${EVAL_BATCH_SIZE}" "${MAX_SAMPLES_PER_TASK}" "${MODEL_DEVICE_MAP}"
import json
import shutil
import sys
from pathlib import Path

from florence_forge.cli.commands_eval import _build_eval_dataset_from_jsonl
from florence_forge.core.config import ModelConfig
from florence_forge.core.model import Florence2MultiTaskModel
from florence_forge.evaluation.evaluator import MultiTaskEvaluator

model_path = sys.argv[1]
data_path = sys.argv[2]
device = sys.argv[3]
output_path = Path(sys.argv[4])
batch_size = int(sys.argv[5])
max_samples_per_task = int(sys.argv[6])
device_map = sys.argv[7]

model = Florence2MultiTaskModel(
    ModelConfig(
        model_name=model_path,
        device=device,
        device_map=device_map,
        use_lora=False,
    )
)
model.load()

dataset = _build_eval_dataset_from_jsonl(data_path, model)
evaluator = MultiTaskEvaluator(model, device=device)
results = evaluator.evaluate_dataset(
    dataset,
    batch_size=batch_size,
    num_workers=0,
    max_samples_per_task=max_samples_per_task,
)

payload = {
    "model_path": model_path,
    "data_path": data_path,
    "device": device,
    "batch_size": batch_size,
    "overall_metrics": results.get("overall_metrics", {}),
    "task_metrics": results.get("task_metrics", {}),
    "evaluation_info": {
        "total_samples": results.get("total_samples"),
        "num_tasks": results.get("num_tasks"),
        "max_samples_per_task": max_samples_per_task,
    },
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

temp_dir = getattr(dataset, "_eval_temp_dir", None)
if temp_dir:
    shutil.rmtree(temp_dir, ignore_errors=True)

print("Saved:", output_path)
PY

echo "[3/3] Build comparison summary"
"${PYTHON_BIN}" - <<'PY' "${BASE_JSON}" "${FINETUNED_JSON}" "${COMPARE_JSON}"
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
finetuned_path = Path(sys.argv[2])
compare_path = Path(sys.argv[3])

with base_path.open("r", encoding="utf-8") as f:
    base = json.load(f)
with finetuned_path.open("r", encoding="utf-8") as f:
    finetuned = json.load(f)

base_metrics = base.get("overall_metrics", {})
finetuned_metrics = finetuned.get("overall_metrics", {})

metric_names = sorted(set(base_metrics) | set(finetuned_metrics))
diff = {}
for name in metric_names:
    base_value = base_metrics.get(name)
    finetuned_value = finetuned_metrics.get(name)
    if isinstance(base_value, (int, float)) and isinstance(finetuned_value, (int, float)):
        diff[name] = finetuned_value - base_value

summary = {
    "data": base.get("evaluation_info", {}).get("total_samples"),
    "base_result_file": str(base_path),
    "finetuned_result_file": str(finetuned_path),
    "base_overall_metrics": base_metrics,
    "finetuned_overall_metrics": finetuned_metrics,
    "delta_finetuned_minus_base": diff,
}

with compare_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Comparison saved to:", compare_path)
for key in ("avg_mAP", "avg_precision", "avg_recall", "avg_f1"):
    b = base_metrics.get(key)
    ft = finetuned_metrics.get(key)
    if isinstance(b, (int, float)) and isinstance(ft, (int, float)):
        print(f"{key}: base={b:.6f} finetuned={ft:.6f} delta={ft - b:+.6f}")
PY

echo "Done."
echo "Eval batch size:  ${EVAL_BATCH_SIZE}"
echo "Max samples/task: ${MAX_SAMPLES_PER_TASK}"
echo "Model device_map: ${MODEL_DEVICE_MAP}"
echo "Base result:      ${BASE_JSON}"
echo "Finetuned result: ${FINETUNED_JSON}"
echo "Compare result:   ${COMPARE_JSON}"
