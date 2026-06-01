"""评估器单元测试。"""

import json

import torch

from florence_forge.evaluation.evaluator import MultiTaskEvaluator


class DummyModel:
    def __init__(self):
        self.processor = object()

    def generate(self, *args, **kwargs):
        return None

    def to(self, device):
        return self


class DecodeOnlyModel:
    def __init__(self):
        self.decoded = None

    def generate(self, *args, **kwargs):
        return torch.tensor([[1, 2]])

    def decode(self, token_ids, skip_special_tokens=True):
        self.decoded = (token_ids, skip_special_tokens)
        return ["decoded text"]

    def to(self, device):
        return self


class ProcessorBackedDataset:
    processor = object()
    backend = None
    collate_fn = object()


def test_clean_prediction_removes_florence_prompt():
    evaluator = MultiTaskEvaluator(DummyModel(), device="cpu")
    cleaned = evaluator._clean_prediction("<CAPTION> a cat on a sofa", "CAPTION")
    assert cleaned == "a cat on a sofa"


def test_clean_reference_removes_florence_prompt():
    evaluator = MultiTaskEvaluator(DummyModel(), device="cpu")
    cleaned = evaluator._clean_reference("<OCR> hello world", "OCR")
    assert cleaned == "hello world"


def test_extract_generated_tokens_removes_only_matching_prompt_prefix():
    evaluator = MultiTaskEvaluator(DummyModel(), device="cpu")
    input_ids = torch.tensor([[1, 2, 3]])

    full_sequence = torch.tensor([[1, 2, 3, 4, 5]])
    new_only = torch.tensor([[4, 5]])
    longer_without_prompt = torch.tensor([[9, 9, 9, 4, 5]])

    assert evaluator._extract_generated_tokens(full_sequence, input_ids).tolist() == [[4, 5]]
    assert evaluator._extract_generated_tokens(new_only, input_ids).tolist() == [[4, 5]]
    assert evaluator._extract_generated_tokens(longer_without_prompt, input_ids).tolist() == [[9, 9, 9, 4, 5]]


def test_evaluator_accepts_model_decode_without_processor():
    model = DecodeOnlyModel()
    evaluator = MultiTaskEvaluator(model, device="cpu")

    tokens = torch.tensor([[1, 2, 3]])
    decoded = evaluator._decode_token_ids(tokens)

    assert decoded == ["decoded text"]
    assert model.decoded == (tokens, True)


def test_evaluator_disables_workers_for_processor_backed_dataset():
    evaluator = MultiTaskEvaluator(DummyModel(), device="cpu")

    num_workers = evaluator._resolve_num_workers(ProcessorBackedDataset(), 4)

    assert num_workers == 0


def test_export_bad_cases_writes_thresholded_jsonl(tmp_path):
    evaluator = MultiTaskEvaluator(DummyModel(), device="cpu")
    results = [
        {"sample_id": "a", "task_type": "CAPTION", "prediction": "cat", "reference": "dog", "score": 0.2},
        {"sample_id": "b", "task_type": "CAPTION", "prediction": "cat", "reference": "cat", "score": 0.9},
        {"sample_id": "c", "task_type": "OCR", "prediction": "helo", "reference": "hello"},
    ]

    output_file = evaluator.export_bad_cases(results, threshold=0.5, output_dir=tmp_path)

    exported = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]

    assert [item["sample_id"] for item in exported] == ["a", "c"]
    assert exported[0]["score"] == 0.2
    assert exported[1]["score"] == 0.0


def test_export_bad_cases_accepts_evaluate_task_result_dict(tmp_path):
    evaluator = MultiTaskEvaluator(DummyModel(), device="cpu")
    result = {
        "task_type": "CAPTION",
        "predictions": ["a cat", "a dog"],
        "references": ["a cat", "a bird"],
    }

    output_file = evaluator.export_bad_cases(result, threshold=0.5, output_dir=tmp_path)

    exported = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]

    assert len(exported) == 1
    assert exported[0]["prediction"] == "a dog"
    assert exported[0]["task_type"] == "CAPTION"
