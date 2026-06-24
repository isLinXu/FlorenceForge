"""Unit tests for pure-Python Agentic meta-cognitive components.

These tests validate the agentic token system, chain builders, and reward
model logic WITHOUT requiring torch or the full FlorenceForge model stack.
They cover the integration touchpoints identified in the audit:

  1. Token registration & phase helpers (agentic_tokens.py)
  2. Chain builder output structure (agentic_trajectory_expander.py)
  3. Reward model scoring logic (reward_models.py — agentic models only)
  4. Task config validation (tasks.py — is_agentic flag)
  5. TVP task alias routing (tvp_training.py — agentic aliases)
"""

from __future__ import annotations


import pytest

# ---------------------------------------------------------------------------
# Fixtures: ensure we can import the pure-Python modules
# ---------------------------------------------------------------------------

# These modules have no heavy torch deps for the functions we test:
# - agentic_tokens: pure Python
# - tasks: pure Python (Pydantic only)
# - tvp_training aliases: pure dict (we only test the alias dict)


# ---------------------------------------------------------------------------
# 1. agentic_tokens.py — token definitions & helpers
# ---------------------------------------------------------------------------

class TestAgenticTokens:
    """Test the token vocabulary and text helper functions."""

    def test_token_pairs_are_well_formed(self):
        from florence_forge.core.agentic_tokens import AGENTIC_TOKEN_PAIRS

        for open_tok, close_tok in AGENTIC_TOKEN_PAIRS:
            assert open_tok.startswith("<") and open_tok.endswith(">")
            assert close_tok.startswith("</") and close_tok.endswith(">")
            # close token must be open token with a slash
            assert close_tok == "</" + open_tok[1:]

    def test_special_tokens_flat_list(self):
        from florence_forge.core.agentic_tokens import (
            AGENTIC_SPECIAL_TOKENS, AGENTIC_TOKEN_PAIRS,
        )

        expected = [tok for pair in AGENTIC_TOKEN_PAIRS for tok in pair]
        assert AGENTIC_SPECIAL_TOKENS == expected
        assert len(AGENTIC_SPECIAL_TOKENS) == 14  # 7 pairs

    def test_phase_order_canonical(self):
        from florence_forge.core.agentic_tokens import AGENTIC_PHASE_ORDER

        assert "plan" in AGENTIC_PHASE_ORDER
        assert "act" in AGENTIC_PHASE_ORDER
        assert "verify" in AGENTIC_PHASE_ORDER
        assert "reflect" in AGENTIC_PHASE_ORDER
        assert "decide" in AGENTIC_PHASE_ORDER
        assert "summarize_state" in AGENTIC_PHASE_ORDER
        assert "done" in AGENTIC_PHASE_ORDER

    def test_required_phases_subset(self):
        from florence_forge.core.agentic_tokens import (
            REQUIRED_PHASES, AGENTIC_PHASE_ORDER,
        )

        for phase in REQUIRED_PHASES:
            assert phase in AGENTIC_PHASE_ORDER

    def test_wrap_phase(self):
        from florence_forge.core.agentic_tokens import wrap_phase

        assert wrap_phase("plan", "strategy") == "<PLAN>strategy</PLAN>"
        assert wrap_phase("act", "scan") == "<ACT>scan</ACT>"
        assert wrap_phase("verify", "check") == "<VERIFY>check</VERIFY>"
        assert wrap_phase("reflect", "error found") == "<REFLECT>error found</REFLECT>"
        assert wrap_phase("decide", "final answer") == "<DECIDE>final answer</DECIDE>"

    def test_wrap_phase_unknown_raises(self):
        from florence_forge.core.agentic_tokens import wrap_phase

        with pytest.raises(ValueError, match="Unknown agentic phase"):
            wrap_phase("unknown_phase", "content")

    def test_extract_phase_single(self):
        from florence_forge.core.agentic_tokens import extract_phase

        text = "<PLAN>scan left to right</PLAN>"
        result = extract_phase(text, "plan")
        assert result == ["scan left to right"]

    def test_extract_phase_multiple(self):
        from florence_forge.core.agentic_tokens import extract_phase

        text = "<ACT>step1</ACT> middle <ACT>step2</ACT>"
        result = extract_phase(text, "act")
        assert result == ["step1", "step2"]

    def test_extract_phase_none(self):
        from florence_forge.core.agentic_tokens import extract_phase

        text = "no tokens here"
        result = extract_phase(text, "plan")
        assert result == []

    def test_extract_all_phases(self):
        from florence_forge.core.agentic_tokens import extract_all_phases

        text = (
            "<PLAN>strategy</PLAN>"
            "<ACT>action</ACT>"
            "<VERIFY>ok</VERIFY>"
            "<DECIDE>answer</DECIDE>"
        )
        result = extract_all_phases(text)
        assert result["plan"] == ["strategy"]
        assert result["act"] == ["action"]
        assert result["verify"] == ["ok"]
        assert result["decide"] == ["answer"]
        assert result["reflect"] == []  # not present

    def test_has_required_phases_true(self):
        from florence_forge.core.agentic_tokens import has_required_phases

        text = "<ACT>do something</ACT> <DECIDE>answer</DECIDE>"
        assert has_required_phases(text) is True

    def test_has_required_phases_false_missing_decide(self):
        from florence_forge.core.agentic_tokens import has_required_phases

        text = "<ACT>do something</ACT>"
        assert has_required_phases(text) is False

    def test_has_required_phases_false_missing_act(self):
        from florence_forge.core.agentic_tokens import has_required_phases

        text = "<DECIDE>answer</DECIDE>"
        assert has_required_phases(text) is False

    def test_get_phase_order(self):
        from florence_forge.core.agentic_tokens import get_phase_order

        text = "<ACT>step1</ACT><VERIFY>check</VERIFY><PLAN>replan</PLAN><DECIDE>done</DECIDE>"
        order = get_phase_order(text)
        # get_phase_order returns phases in canonical order, not text order
        assert "act" in order
        assert "verify" in order
        assert "plan" in order
        assert "decide" in order
        assert "reflect" not in order

    def test_register_agentic_tokens_none(self):
        from florence_forge.core.agentic_tokens import register_agentic_tokens

        assert register_agentic_tokens(None) == 0

    def test_register_agentic_tokens_with_fake_tokenizer(self):
        from florence_forge.core.agentic_tokens import (
            register_agentic_tokens, AGENTIC_SPECIAL_TOKENS,
        )

        class FakeTokenizer:
            def __init__(self):
                self._vocab = {"<s>": 0, "</s>": 1, "hello": 2}
                self.added = []

            def get_vocab(self):
                return dict(self._vocab)

            def add_tokens(self, tokens, special_tokens=False):
                for t in tokens:
                    self._vocab[t] = len(self._vocab)
                self.added.extend(tokens)
                return len(tokens)

        tk = FakeTokenizer()
        added = register_agentic_tokens(tk)
        assert added == len(AGENTIC_SPECIAL_TOKENS)
        for t in AGENTIC_SPECIAL_TOKENS:
            assert t in tk.get_vocab()

        # Second call should be idempotent
        added2 = register_agentic_tokens(tk)
        assert added2 == 0


# ---------------------------------------------------------------------------
# 2. tasks.py — agentic task config validation
# ---------------------------------------------------------------------------

class TestAgenticTaskConfigs:
    """Test that agentic tasks are properly registered in the task registry."""

    def test_agentic_task_names_exist(self):
        from florence_forge.core.tasks import FLORENCE2_TASKS

        expected = {"AGENTIC_COUNT", "AGENTIC_SPATIAL", "AGENTIC_MAZE", "AGENTIC_GROUNDING"}
        assert expected.issubset(set(FLORENCE2_TASKS.keys()))

    def test_agentic_tasks_have_is_agentic_flag(self):
        from florence_forge.core.tasks import FLORENCE2_TASKS

        for name, cfg in FLORENCE2_TASKS.items():
            if name.startswith("AGENTIC_"):
                assert cfg.is_agentic is True, f"{name} should have is_agentic=True"

    def test_non_agentic_tasks_not_flagged(self):
        from florence_forge.core.tasks import FLORENCE2_TASKS

        for name, cfg in FLORENCE2_TASKS.items():
            if not name.startswith("AGENTIC_"):
                assert cfg.is_agentic is False, f"{name} should have is_agentic=False"

    def test_is_agentic_task_helper(self):
        from florence_forge.core.tasks import is_agentic_task

        assert is_agentic_task("AGENTIC_COUNT") is True
        assert is_agentic_task("AGENTIC_MAZE") is True
        assert is_agentic_task("OD") is False
        assert is_agentic_task("CAPTION") is False

    def test_get_agentic_tasks(self):
        from florence_forge.core.tasks import get_agentic_tasks

        tasks = get_agentic_tasks()
        assert len(tasks) == 4
        for name, cfg in tasks.items():
            assert cfg.is_agentic is True

    def test_agentic_tasks_are_visual_primitive(self):
        """Agentic tasks share the visual-primitive pipeline but are NOT TVP CoT tasks."""
        from florence_forge.core.tasks import FLORENCE2_TASKS

        for name, cfg in FLORENCE2_TASKS.items():
            if cfg.is_agentic:
                assert cfg.is_visual_primitive is True, \
                    f"{name} should have is_visual_primitive=True"
                assert cfg.is_tvp is False, \
                    f"{name} should NOT be is_tvp=True (agentic ≠ TVP CoT)"

    def test_agentic_task_prompts_are_florence2_tokens(self):
        """Agentic task prompts must be valid Florence-2 task tokens, not conversational text."""
        from florence_forge.core.tasks import FLORENCE2_TASKS

        for name, cfg in FLORENCE2_TASKS.items():
            if cfg.is_agentic:
                prompt = cfg.prompt
                assert prompt.startswith("<") and prompt.endswith(">"), \
                    f"{name} prompt '{prompt}' must be a Florence-2 task token"


# ---------------------------------------------------------------------------
# 3. tvp_training.py — alias routing
# ---------------------------------------------------------------------------

class TestTVPTaskAliases:
    """Test that TVP YAML aliases correctly route agentic tasks."""

    def test_agentic_aliases_exist(self):
        from florence_forge.training.tvp_training import TVP_TASK_ALIASES

        assert TVP_TASK_ALIASES["agentic_count"] == "AGENTIC_COUNT"
        assert TVP_TASK_ALIASES["agentic_spatial"] == "AGENTIC_SPATIAL"
        assert TVP_TASK_ALIASES["agentic_maze"] == "AGENTIC_MAZE"
        assert TVP_TASK_ALIASES["agentic_grounding"] == "AGENTIC_GROUNDING"

    def test_normalize_tvp_task_type_agentic(self):
        from florence_forge.training.tvp_training import normalize_tvp_task_type

        assert normalize_tvp_task_type("agentic_count") == "AGENTIC_COUNT"
        assert normalize_tvp_task_type("AGENTIC_COUNT") == "AGENTIC_COUNT"

    def test_normalize_tvp_task_type_standard(self):
        from florence_forge.training.tvp_training import normalize_tvp_task_type

        assert normalize_tvp_task_type("od") == "OD"
        assert normalize_tvp_task_type("caption") == "CAPTION"

    def test_normalize_tvp_task_type_empty_raises(self):
        from florence_forge.training.tvp_training import normalize_tvp_task_type

        with pytest.raises(ValueError, match="must not be empty"):
            normalize_tvp_task_type("")

    def test_apply_mixed_training_weights_includes_agentic(self):
        from florence_forge.training.tvp_training import apply_mixed_training_weights

        data_configs = [
            {"task_type": "OD", "weight": 1.0},
            {"task_type": "AGENTIC_COUNT", "weight": 1.0},
            {"task_type": "CAPTION", "weight": 1.0},
        ]
        result = apply_mixed_training_weights(data_configs, tvp_ratio=0.3)
        # agentic task should be in tvp_items and get scaled
        agentic_item = next(r for r in result if r["task_type"] == "AGENTIC_COUNT")
        assert agentic_item["weight"] > 0


# ---------------------------------------------------------------------------
# 4. config.py — LoRAConfig BART defaults
# ---------------------------------------------------------------------------

class TestLoRAConfigBART:
    """Test that LoRAConfig defaults are correct for BART architecture."""

    def test_default_target_modules_bart(self):
        from florence_forge.core.config import LoRAConfig

        cfg = LoRAConfig()
        # BART uses fc1/fc2, not gate_proj/up_proj/down_proj
        assert "fc1" in cfg.target_modules
        assert "fc2" in cfg.target_modules
        assert "q_proj" in cfg.target_modules
        assert "k_proj" in cfg.target_modules
        assert "v_proj" in cfg.target_modules
        assert "o_proj" in cfg.target_modules
        # Should NOT have LLaMA-style modules by default
        assert "gate_proj" not in cfg.target_modules
        assert "up_proj" not in cfg.target_modules
        assert "down_proj" not in cfg.target_modules

    def test_modules_to_save_default(self):
        from florence_forge.core.config import LoRAConfig

        cfg = LoRAConfig()
        assert "lm_head" in cfg.modules_to_save
        assert "embed_tokens" in cfg.modules_to_save

    def test_modules_to_save_customizable(self):
        from florence_forge.core.config import LoRAConfig

        cfg = LoRAConfig(modules_to_save=[])
        assert cfg.modules_to_save == []


# ---------------------------------------------------------------------------
# 5. config.py — ModelConfig enable_agentic_tokens
# ---------------------------------------------------------------------------

class TestModelConfigAgentic:
    """Test that ModelConfig has the enable_agentic_tokens field."""

    def test_enable_agentic_tokens_default_false(self):
        from florence_forge.core.config import ModelConfig

        cfg = ModelConfig()
        assert cfg.enable_agentic_tokens is False

    def test_enable_agentic_tokens_can_be_set(self):
        from florence_forge.core.config import ModelConfig

        cfg = ModelConfig(enable_agentic_tokens=True)
        assert cfg.enable_agentic_tokens is True

    def test_enable_visual_primitives_still_works(self):
        from florence_forge.core.config import ModelConfig

        cfg = ModelConfig(enable_visual_primitives=True, enable_agentic_tokens=True)
        assert cfg.enable_visual_primitives is True
        assert cfg.enable_agentic_tokens is True


# ---------------------------------------------------------------------------
# 6. agentic_synthetic.py — chain structure validation (no torch needed)
# ---------------------------------------------------------------------------

class TestAgenticChainBuilderStructure:
    """Test that AgenticChainBuilder produces well-formed agentic chains.

    These tests only check the text structure (token presence) without
    requiring the full TVP converter pipeline.
    """

    def test_counting_chain_structure(self):
        from florence_forge.data.agentic_trajectory_expander import AgenticChainBuilder
        from florence_forge.core.agentic_tokens import has_required_phases

        chain = AgenticChainBuilder.build_counting_chain(
            label="box",
            boxes=[[10, 20, 50, 60], [70, 80, 100, 110]],
            count=2,
        )
        assert has_required_phases(chain) is True
        assert "<PLAN>" in chain and "</PLAN>" in chain
        assert "<ACT>" in chain and "</ACT>" in chain
        assert "<VERIFY>" in chain and "</VERIFY>" in chain
        assert "<DECIDE>" in chain and "</DECIDE>" in chain

    def test_counting_chain_with_error_injection(self):
        from florence_forge.data.agentic_trajectory_expander import AgenticChainBuilder
        from florence_forge.core.agentic_tokens import extract_phase

        chain = AgenticChainBuilder.build_counting_chain(
            label="box",
            boxes=[[10, 20, 50, 60]],
            count=1,
            inject_error=True,
        )
        # Error injection should introduce a REFLECT phase
        assert "<REFLECT>" in chain
        reflect_content = extract_phase(chain, "reflect")
        assert len(reflect_content) > 0

    def test_spatial_chain_structure(self):
        from florence_forge.data.agentic_trajectory_expander import AgenticChainBuilder
        from florence_forge.core.agentic_tokens import has_required_phases

        chain = AgenticChainBuilder.build_spatial_chain(
            observation="Red box is left of blue box",
            reasoning="Red is at x=10, blue at x=100, so red is left",
            answer="left",
            supporting_boxes={"red box": [[10, 20, 50, 60]], "blue box": [[70, 20, 110, 60]]},
        )
        assert has_required_phases(chain) is True

    def test_maze_chain_structure(self):
        from florence_forge.data.agentic_trajectory_expander import AgenticChainBuilder
        from florence_forge.core.agentic_tokens import has_required_phases

        chain = AgenticChainBuilder.build_maze_chain(
            solvable=True,
            exploration_points=[(0, 0), (1, 0), (2, 0)],
            solution_points=[(0, 0), (1, 0), (2, 0)],
            answer="true",
            start_point=(0, 0),
            end_point=(2, 0),
            exploration_steps=[{"points": [(0, 0)], "note": "start"}],
        )
        assert has_required_phases(chain) is True

    def test_grounding_chain_structure(self):
        from florence_forge.data.agentic_trajectory_expander import AgenticChainBuilder
        from florence_forge.core.agentic_tokens import has_required_phases

        chain = AgenticChainBuilder.build_grounding_chain(
            caption="Find the red block",
            label="red block",
            boxes=[[10, 20, 50, 60]],
        )
        assert has_required_phases(chain) is True


# ---------------------------------------------------------------------------
# 7. reward_models.py — agentic reward model scoring (no torch needed)
# ---------------------------------------------------------------------------

class TestAgenticRewardModels:
    """Test agentic reward model scoring logic."""

    def test_agentic_format_reward_model_well_formed(self):
        from florence_forge.training.reward_models import AgenticFormatRewardModel
        from florence_forge.core.agentic_tokens import wrap_phase

        rm = AgenticFormatRewardModel()
        # Well-formed chain with all required phases
        good_chain = (
            wrap_phase("plan", "strategy")
            + wrap_phase("act", "action")
            + wrap_phase("verify", "check")
            + wrap_phase("decide", "answer")
        )
        score = rm(good_chain)
        assert score > 0.5, f"Well-formed chain should score high, got {score}"

    def test_agentic_format_reward_model_missing_phases(self):
        from florence_forge.training.reward_models import AgenticFormatRewardModel

        rm = AgenticFormatRewardModel()
        # Missing ACT and DECIDE
        bad_chain = "<PLAN>strategy</PLAN>"
        score = rm(bad_chain)
        # Partial credit for properly closed tags, but lower than well-formed
        assert score < 1.0, f"Malformed chain should not get full score, got {score}"

    def test_agentic_format_reward_model_empty(self):
        from florence_forge.training.reward_models import AgenticFormatRewardModel

        rm = AgenticFormatRewardModel()
        score = rm("")
        # Empty text gets partial base credit but should not exceed well-formed
        assert score < 1.0

    def test_build_agentic_reward_models_count(self):
        from florence_forge.training.reward_models import build_agentic_reward_models

        models = build_agentic_reward_models()
        assert len(models) >= 3, "Should have at least 3 agentic reward models"


# ---------------------------------------------------------------------------
# 8. New tokens: SUMMARIZE_STATE, DONE, phase loss weights, find_phase_spans
# ---------------------------------------------------------------------------

class TestExtendedAgenticTokens:
    """Test the extended agentic token system (SUMMARIZE_STATE, DONE, etc.)."""

    def test_summarize_state_token_exists(self):
        from florence_forge.core.agentic_tokens import AGENTIC_PHASE_TOKENS

        assert "summarize_state" in AGENTIC_PHASE_TOKENS
        open_tok, close_tok = AGENTIC_PHASE_TOKENS["summarize_state"]
        assert open_tok == "<SUMMARIZE_STATE>"
        assert close_tok == "</SUMMARIZE_STATE>"

    def test_done_token_exists(self):
        from florence_forge.core.agentic_tokens import AGENTIC_PHASE_TOKENS

        assert "done" in AGENTIC_PHASE_TOKENS
        open_tok, close_tok = AGENTIC_PHASE_TOKENS["done"]
        assert open_tok == "<DONE>"
        assert close_tok == "</DONE>"

    def test_wrap_phase_summarize_state(self):
        from florence_forge.core.agentic_tokens import wrap_phase

        result = wrap_phase("summarize_state", "detected 5 objects")
        assert result == "<SUMMARIZE_STATE>detected 5 objects</SUMMARIZE_STATE>"

    def test_wrap_phase_done(self):
        from florence_forge.core.agentic_tokens import wrap_phase

        result = wrap_phase("done", "Task completed.")
        assert result == "<DONE>Task completed.</DONE>"

    def test_phase_loss_weights_exist(self):
        from florence_forge.core.agentic_tokens import PHASE_LOSS_WEIGHTS

        assert "plan" in PHASE_LOSS_WEIGHTS
        assert "act" in PHASE_LOSS_WEIGHTS
        assert "verify" in PHASE_LOSS_WEIGHTS
        assert "reflect" in PHASE_LOSS_WEIGHTS
        assert "decide" in PHASE_LOSS_WEIGHTS
        assert "summarize_state" in PHASE_LOSS_WEIGHTS
        assert "done" in PHASE_LOSS_WEIGHTS
        # DECIDE should have the highest weight
        assert PHASE_LOSS_WEIGHTS["decide"] >= PHASE_LOSS_WEIGHTS["act"]
        # REFLECT should be boosted for self-correction
        assert PHASE_LOSS_WEIGHTS["reflect"] > PHASE_LOSS_WEIGHTS["plan"]

    def test_get_phase_loss_weights_returns_copy(self):
        from florence_forge.core.agentic_tokens import (
            get_phase_loss_weights, PHASE_LOSS_WEIGHTS,
        )

        weights = get_phase_loss_weights()
        assert weights == PHASE_LOSS_WEIGHTS
        weights["plan"] = 999.0
        # Original should be unchanged
        from florence_forge.core.agentic_tokens import PHASE_LOSS_WEIGHTS as orig
        assert orig["plan"] != 999.0

    def test_find_phase_spans(self):
        from florence_forge.core.agentic_tokens import find_phase_spans

        text = "<PLAN>strategy</PLAN><ACT>action</ACT><DECIDE>answer</DECIDE>"
        spans = find_phase_spans(text)
        assert len(spans) == 3
        phases = [s[0] for s in spans]
        assert "plan" in phases
        assert "act" in phases
        assert "decide" in phases
        # Spans should be sorted by position
        assert spans[0][1] < spans[1][1] < spans[2][1]

    def test_find_phase_spans_empty(self):
        from florence_forge.core.agentic_tokens import find_phase_spans

        assert find_phase_spans("no tokens here") == []

    def test_core_and_control_phases(self):
        from florence_forge.core.agentic_tokens import CORE_PHASES, CONTROL_PHASES

        assert "plan" in CORE_PHASES
        assert "summarize_state" in CONTROL_PHASES
        assert "done" in CONTROL_PHASES
        assert "act" not in CONTROL_PHASES


# ---------------------------------------------------------------------------
# 9. Multi-round chain builder
# ---------------------------------------------------------------------------

class TestMultiRoundChainBuilder:
    """Test the build_multi_round_chain method."""

    def test_multi_round_basic_structure(self):
        from florence_forge.data.agentic_trajectory_expander import AgenticChainBuilder
        from florence_forge.core.agentic_tokens import has_required_phases

        rounds = [
            {"plan": "Round 1: detect objects", "act": "Running <OD>", "verify": "Found 3 objects"},
            {"plan": "Round 2: extract text", "act": "Running <OCR>", "verify": "Text extracted"},
        ]
        chain = AgenticChainBuilder.build_multi_round_chain(
            rounds=rounds,
            final_answer="3 objects with text",
        )
        assert has_required_phases(chain) is True
        assert "<PLAN>" in chain
        assert "<ACT>" in chain
        assert "<VERIFY>" in chain
        assert "<DECIDE>" in chain
        assert "<DONE>" in chain

    def test_multi_round_has_summarize_state(self):
        from florence_forge.data.agentic_trajectory_expander import AgenticChainBuilder

        rounds = [
            {"plan": "Round 1", "act": "act1", "verify": "verify1"},
            {"plan": "Round 2", "act": "act2", "verify": "verify2"},
        ]
        chain = AgenticChainBuilder.build_multi_round_chain(
            rounds=rounds,
            final_answer="done",
        )
        # Multi-round should include SUMMARIZE_STATE
        assert "<SUMMARIZE_STATE>" in chain

    def test_multi_round_error_injection(self):
        from florence_forge.data.agentic_trajectory_expander import AgenticChainBuilder

        rounds = [
            {"plan": "Round 1", "act": "correct act", "verify": "verified",
             "error_act": "wrong act", "error_verify": "error found",
             "reflect": "I made an error"},
            {"plan": "Round 2", "act": "act2", "verify": "verify2"},
        ]
        chain = AgenticChainBuilder.build_multi_round_chain(
            rounds=rounds,
            final_answer="corrected answer",
            inject_error_at=0,
        )
        assert "<REFLECT>" in chain

    def test_multi_round_done_token(self):
        from florence_forge.data.agentic_trajectory_expander import AgenticChainBuilder

        chain = AgenticChainBuilder.build_multi_round_chain(
            rounds=[{"plan": "p", "act": "a", "verify": "v"}],
            final_answer="answer",
        )
        assert "<DONE>" in chain
        assert "</DONE>" in chain

    def test_wrap_done_helper(self):
        from florence_forge.data.agentic_trajectory_expander import wrap_done

        result = wrap_done()
        assert "<DONE>" in result
        assert "</DONE>" in result

    def test_wrap_summarize_state_helper(self):
        from florence_forge.data.agentic_trajectory_expander import wrap_summarize_state

        result = wrap_summarize_state("5 objects detected")
        assert "<SUMMARIZE_STATE>5 objects detected</SUMMARIZE_STATE>" == result


# ---------------------------------------------------------------------------
# 10. AgenticEvaluator metrics
# ---------------------------------------------------------------------------

class TestAgenticEvaluator:
    """Test the agentic evaluation metrics."""

    def test_format_validity_well_formed(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_format_validity

        text = (
            "<PLAN>plan</PLAN>"
            "<ACT>action</ACT>"
            "<VERIFY>verify</VERIFY>"
            "<DECIDE>answer</DECIDE>"
        )
        is_valid, issues = evaluate_format_validity(text)
        assert is_valid is True
        assert len(issues) == 0

    def test_format_validity_missing_phases(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_format_validity

        text = "<PLAN>plan</PLAN>"
        is_valid, issues = evaluate_format_validity(text)
        assert is_valid is False
        assert len(issues) > 0

    def test_planning_accuracy_with_native_prompts(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_planning_accuracy

        plan = "I will use <OD> to detect objects and <OCR> to read text"
        score = evaluate_planning_accuracy(plan)
        assert score > 0.5

    def test_planning_accuracy_empty(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_planning_accuracy

        assert evaluate_planning_accuracy("") == 0.0

    def test_tool_call_correctness_with_native_prompt(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_tool_call_correctness

        act = "Running <OD> to detect objects"
        score = evaluate_tool_call_correctness(act)
        assert score >= 0.5

    def test_tool_call_correctness_with_vp_box_marker(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_tool_call_correctness

        act = "Found red block at <|box|>[[100,200,300,400]]<|/box|>"
        score = evaluate_tool_call_correctness(act)
        assert score >= 0.5, f"VP box marker should score >= 0.5 (got {score})"

    def test_tool_call_correctness_with_vp_point_marker(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_tool_call_correctness

        act = "Start: <|point|>[[167,167]]<|/point|>"
        score = evaluate_tool_call_correctness(act)
        assert score >= 0.5, f"VP point marker should score >= 0.5 (got {score})"

    def test_tool_call_correctness_empty(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_tool_call_correctness

        assert evaluate_tool_call_correctness("") == 0.0

    def test_error_recovery_detected(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_error_recovery

        text = (
            "<ACT>counted 3</ACT>"
            "<VERIFY>Wait, I missed one. There are actually 4.</VERIFY>"
            "<REFLECT>I missed the rightmost object during scanning.</REFLECT>"
            "<DECIDE>The correct count is 4.</DECIDE>"
        )
        score = evaluate_error_recovery(text, {"error_injected": True})
        assert score > 0.5

    def test_error_recovery_no_error(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_error_recovery

        text = (
            "<ACT>counted 4</ACT>"
            "<VERIFY>Verified: 4 objects confirmed.</VERIFY>"
            "<DECIDE>The count is 4.</DECIDE>"
        )
        score = evaluate_error_recovery(text, {"error_injected": False})
        # No error detected, but that's fine for non-injected samples
        assert score >= 0.0

    def test_consistency_single_round(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_consistency

        assert evaluate_consistency("text", num_rounds=1) == 1.0

    def test_native_preservation(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_native_preservation

        # 90% retention
        score = evaluate_native_preservation(0.80, 0.72)
        assert abs(score - 0.9) < 0.01

    def test_native_preservation_zero_before(self):
        from florence_forge.evaluation.agentic_evaluator import evaluate_native_preservation

        assert evaluate_native_preservation(0.0, 0.0) == 1.0

    def test_agentic_evaluator_end_to_end(self):
        from florence_forge.evaluation.agentic_evaluator import AgenticEvaluator

        evaluator = AgenticEvaluator()
        evaluator.add_sample(
            "<PLAN>plan</PLAN><ACT><OD></ACT><VERIFY>ok</VERIFY><DECIDE>answer</DECIDE>",
            {"error_injected": False, "num_rounds": 1},
        )
        evaluator.add_sample(
            "<PLAN>plan</PLAN><ACT><OCR></ACT><VERIFY>ok</VERIFY><DECIDE>answer</DECIDE>",
            {"error_injected": True, "num_rounds": 1},
        )
        evaluator.set_native_accuracy(0.80, 0.76)

        metrics = evaluator.compute()
        assert metrics.total_samples == 2
        assert metrics.error_injected_samples == 1
        assert metrics.format_validity > 0
        assert metrics.native_capability_preservation > 0.9

    def test_agentic_metrics_summary(self):
        from florence_forge.evaluation.agentic_evaluator import AgenticMetrics

        m = AgenticMetrics(
            format_validity=0.9,
            planning_accuracy=0.8,
            total_samples=100,
        )
        summary = m.summary()
        assert "Agentic Evaluation Summary" in summary
        assert "100" in summary

    def test_agentic_metrics_to_dict(self):
        from florence_forge.evaluation.agentic_evaluator import AgenticMetrics

        m = AgenticMetrics(format_validity=0.9, total_samples=50)
        d = m.to_dict()
        assert d["format_validity"] == 0.9
        assert d["total_samples"] == 50


# ---------------------------------------------------------------------------
# 11. Seed task templates and LLM trajectory augmenter
# ---------------------------------------------------------------------------

class TestSeedTasks:
    """Test the seed task template system."""

    def test_seed_tasks_exist(self):
        from florence_forge.data.seed_tasks import SEED_TASKS

        assert len(SEED_TASKS) > 0
        for seed in SEED_TASKS:
            assert seed.task_id
            assert seed.goal
            assert seed.domain

    def test_seed_task_library_get_by_domain(self):
        from florence_forge.data.seed_tasks import SeedTaskLibrary

        lib = SeedTaskLibrary()
        doc_seeds = lib.get_seeds_by_domain("document")
        assert len(doc_seeds) > 0
        for s in doc_seeds:
            assert s.domain == "document"

    def test_seed_task_library_get_by_difficulty(self):
        from florence_forge.data.seed_tasks import SeedTaskLibrary

        lib = SeedTaskLibrary()
        hard_seeds = lib.get_seeds_by_difficulty("hard")
        assert len(hard_seeds) > 0

    def test_seed_task_library_random(self):
        from florence_forge.data.seed_tasks import SeedTaskLibrary

        lib = SeedTaskLibrary()
        seeds = lib.random_seeds(3, seed=42)
        assert len(seeds) == 3

    def test_seed_task_to_dict_roundtrip(self):
        from florence_forge.data.seed_tasks import SeedTaskTemplate

        seed = SeedTaskTemplate(
            task_id="test_001",
            goal="Test goal",
            domain="test",
            expected_steps=["step1", "step2"],
        )
        d = seed.to_dict()
        restored = SeedTaskTemplate.from_dict(d)
        assert restored.task_id == "test_001"
        assert restored.goal == "Test goal"
        assert restored.expected_steps == ["step1", "step2"]

    def test_seed_task_library_json_export_import(self, tmp_path):
        from florence_forge.data.seed_tasks import SeedTaskLibrary

        lib = SeedTaskLibrary()
        json_path = tmp_path / "seeds.json"
        lib.to_json(json_path)
        assert json_path.exists()

        restored = SeedTaskLibrary.from_json(json_path)
        assert len(restored) == len(lib)

    def test_llm_augmenter_validate_trajectory(self):
        from florence_forge.data.seed_tasks import LLMTrajectoryAugmenter

        augmenter = LLMTrajectoryAugmenter(llm_client=None)
        # Valid trajectory
        valid = "<PLAN>p</PLAN><ACT>a</ACT><VERIFY>v</VERIFY><DECIDE>d</DECIDE>"
        assert augmenter._validate_trajectory(valid) is True
        # Invalid (missing ACT and DECIDE)
        invalid = "<PLAN>p</PLAN>"
        assert augmenter._validate_trajectory(invalid) is False

    def test_llm_augmenter_to_jsonl(self, tmp_path):
        from florence_forge.data.seed_tasks import LLMTrajectoryAugmenter

        augmenter = LLMTrajectoryAugmenter(llm_client=None)
        trajectories = [
            {"suffix": "<PLAN>p</PLAN><ACT>a</ACT><VERIFY>v</VERIFY><DECIDE>d</DECIDE>",
             "error_injected": False, "num_rounds": 1, "goal": "test",
             "domain": "test", "expected_steps": ["s1"]},
        ]
        output = augmenter.to_jsonl(trajectories, tmp_path / "out.jsonl")
        assert output.exists()
        import json
        with open(output) as f:
            line = json.loads(f.readline())
        assert "prefix" in line
        assert "suffix" in line
        assert line["agentic"] is True


# ---------------------------------------------------------------------------
# 12. Native task preservation mixer
# ---------------------------------------------------------------------------

class TestNativePreservation:
    """Test the native task preservation data mixer."""

    def test_compute_native_count(self):
        from florence_forge.data.native_preservation import NativeTaskPreserver

        preserver = NativeTaskPreserver(native_ratio=0.3)
        # 70 agentic → 30 native (ratio = 30/100 = 0.3)
        count = preserver.compute_native_count(70)
        assert count == 30

    def test_compute_native_count_zero_ratio(self):
        from florence_forge.data.native_preservation import NativeTaskPreserver

        preserver = NativeTaskPreserver(native_ratio=0.0)
        assert preserver.compute_native_count(100) == 0

    def test_compute_native_count_full_ratio(self):
        from florence_forge.data.native_preservation import NativeTaskPreserver

        preserver = NativeTaskPreserver(native_ratio=1.0)
        assert preserver.compute_native_count(100) == 100

    def test_mix_jsonl_files(self, tmp_path):
        from florence_forge.data.native_preservation import NativeTaskPreserver
        import json

        # Create agentic JSONL
        agentic_path = tmp_path / "agentic.jsonl"
        with open(agentic_path, "w") as f:
            for i in range(10):
                f.write(json.dumps({"prefix": "<COUNT>", "suffix": f"chain_{i}", "agentic": True}) + "\n")

        # Create native JSONL
        native_path = tmp_path / "native.jsonl"
        with open(native_path, "w") as f:
            for i in range(20):
                f.write(json.dumps({"prefix": "<OD>", "suffix": f"box_{i}", "agentic": False}) + "\n")

        output_path = tmp_path / "mixed.jsonl"
        summary = NativeTaskPreserver.mix_jsonl_files(
            agentic_path, native_path, output_path,
            native_ratio=0.3, seed=42,
        )

        assert summary["agentic_count"] == 10
        assert summary["native_count"] > 0
        assert summary["total_count"] == 10 + summary["native_count"]
        assert output_path.exists()

        # Verify output content
        with open(output_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == summary["total_count"]

    def test_native_task_sampler(self, tmp_path):
        from florence_forge.data.native_preservation import NativeTaskSampler
        import json

        # Create a native task file
        task_file = tmp_path / "od.jsonl"
        with open(task_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({"prefix": "<OD>", "suffix": f"box_{i}"}) + "\n")

        sampler = NativeTaskSampler(task_files={"OD": str(task_file)}, seed=42)
        samples = sampler.sample(5)
        assert len(samples) == 5
        for s in samples:
            assert "prefix" in s
            assert "suffix" in s

    def test_native_task_sampler_empty(self):
        from florence_forge.data.native_preservation import NativeTaskSampler

        sampler = NativeTaskSampler()
        assert sampler.sample(10) == []


# ---------------------------------------------------------------------------
# 13. Phase-aware loss weighting
# ---------------------------------------------------------------------------

class TestPhaseAwareLoss:
    """Test the phase-aware loss weighting utilities."""

    def test_build_phase_weight_tensor_basic(self):
        import torch
        from florence_forge.data.phase_aware_loss import build_phase_weight_tensor

        class FakeTokenizer:
            def __init__(self):
                self._vocab = {"<PLAN>": 100, "</PLAN>": 101, "<ACT>": 102,
                               "</ACT>": 103, "<DECIDE>": 104, "</DECIDE>": 105,
                               "<VERIFY>": 106, "</VERIFY>": 107, "hello": 1,
                               "world": 2, "<s>": 0, "</s>": 99}

            def get_vocab(self):
                return dict(self._vocab)

            def convert_tokens_to_ids(self, token):
                return self._vocab.get(token, 0)

        tokenizer = FakeTokenizer()
        labels = torch.tensor([100, 1, 101, 102, 2, 103, 104, 1, 105])
        answer_text = "<PLAN>hello</PLAN><ACT>world</ACT><DECIDE>hello</DECIDE>"

        weights = build_phase_weight_tensor(
            labels=labels,
            answer_text=answer_text,
            tokenizer=tokenizer,
        )
        assert weights.shape == labels.shape
        # All weights should be non-negative
        assert (weights >= 0).all()
        # Phase tokens should have non-zero weights
        assert weights.sum() > 0

    def test_phase_weighted_loss(self):
        import torch
        from florence_forge.data.phase_aware_loss import phase_weighted_loss

        batch_size, seq_len, vocab_size = 2, 5, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[1, 2, 3, -100, 5], [6, 7, 8, 9, -100]])
        weights = torch.ones(batch_size, seq_len)

        loss = phase_weighted_loss(logits, labels, weights)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_phase_weighted_loss_zero_weights(self):
        import torch
        from florence_forge.data.phase_aware_loss import phase_weighted_loss

        logits = torch.randn(1, 3, 5)
        labels = torch.tensor([[1, 2, 3]])
        weights = torch.zeros(1, 3)

        loss = phase_weighted_loss(logits, labels, weights)
        # With zero weights, loss should be near zero (but not NaN)
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# 14. Bug-fix verification tests (from E2E verification)
# ---------------------------------------------------------------------------

class TestBugFixes:
    """Tests for bugs found and fixed during E2E verification."""

    def test_consistency_contradiction_with_coords(self):
        """Contradictory spatial claims with coordinates should be detected."""
        from florence_forge.evaluation.agentic_evaluator import evaluate_consistency
        from florence_forge.core.agentic_tokens import wrap_phase

        # Both rounds have coordinates + spatial contradiction
        multi_bad = (
            wrap_phase("act", "Red box is left of blue box at [100, 200, 300, 400]")
            + wrap_phase("act", "Red box is right of blue box at [100, 200, 300, 400]")
        )
        score = evaluate_consistency(multi_bad, num_rounds=2)
        assert score < 1.0, f"Contradictory spatial with coords should score < 1.0 (got {score})"

    def test_consistency_contradiction_with_shared_objects(self):
        """Contradictory spatial claims with shared object names should be detected."""
        from florence_forge.evaluation.agentic_evaluator import evaluate_consistency
        from florence_forge.core.agentic_tokens import wrap_phase

        multi_bad = (
            wrap_phase("act", "Red box is left of blue box")
            + wrap_phase("act", "Red box is right of blue box")
        )
        score = evaluate_consistency(multi_bad, num_rounds=2)
        assert score < 1.0, f"Contradictory with shared objects should score < 1.0 (got {score})"

    def test_error_recovery_rate_not_over_100_percent(self):
        """Error recovery rate should never exceed 100%."""
        from florence_forge.evaluation.agentic_evaluator import AgenticEvaluator
        from florence_forge.data.agentic_trajectory_expander import AgenticChainBuilder

        evaluator = AgenticEvaluator()

        # Add 5 non-error samples
        for _ in range(5):
            chain = AgenticChainBuilder.build_counting_chain(
                label="box", boxes=[[10, 20, 50, 60]], count=1,
            )
            evaluator.add_sample(chain, {"error_injected": False, "num_rounds": 1})

        # Add 3 error-injected samples
        for _ in range(3):
            chain = AgenticChainBuilder.build_counting_chain(
                label="box", boxes=[[10, 20, 50, 60], [70, 80, 100, 110]],
                count=2, inject_error=True,
            )
            evaluator.add_sample(chain, {"error_injected": True, "num_rounds": 1})

        metrics = evaluator.compute()
        assert metrics.error_recovery_rate <= 1.0, \
            f"Error recovery rate should not exceed 100% (got {metrics.error_recovery_rate:.2%})"

    def test_shares_object_reference_with_nouns(self):
        """_shares_object_reference should detect common significant nouns."""
        from florence_forge.evaluation.agentic_evaluator import _shares_object_reference

        # Both mention "red" and "box"
        assert _shares_object_reference("red box at left", "red box at right") is True
        # Different objects
        assert _shares_object_reference("red box at left", "blue circle at right") is False

    def test_agentic_synthetic_maze_generation(self):
        """agentic_synthetic maze generation should work without width/height args."""
        import random
        from florence_forge.data.agentic_synthetic import generate_agentic_maze

        rng = random.Random(42)
        record, image = generate_agentic_maze(rng, rows=3, cols=3)
        assert "suffix" in record
        assert "prefix" in record
        assert image is not None

    def test_agentic_synthetic_all_types(self):
        """All agentic synthetic generators should work."""
        import random
        from florence_forge.data.agentic_synthetic import (
            generate_agentic_maze,
            generate_agentic_spatial,
            generate_agentic_counting,
            generate_agentic_grounding,
        )

        rng = random.Random(42)
        for gen_func in [generate_agentic_maze, generate_agentic_spatial,
                         generate_agentic_counting, generate_agentic_grounding]:
            record, image = gen_func(rng)
            assert "suffix" in record, f"{gen_func.__name__} missing suffix"
            assert "prefix" in record, f"{gen_func.__name__} missing prefix"
            assert image is not None, f"{gen_func.__name__} returned None image"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
