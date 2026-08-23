"""Gradio WebUI prototype for FlorenceForge Agentic orchestration.

Provides a single-page interface where users upload an image, type a
high-level goal, and watch the multi-step visual reasoning unfold in
real time.

Install dependencies::

    pip install "florence-forge[demo]"

Launch::

    python -m florence_forge.webui.gradio_app \
        --model microsoft/Florence-2-base
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List

from florence_forge.utils.optional_dependencies import missing_dependency_message

logger = logging.getLogger(__name__)

GRADIO_AVAILABLE = False
gr = None  # type: ignore

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    logger.warning(
        missing_dependency_message("Gradio WebUI", "gradio")
    )


# ---------------------------------------------------------------------------
# Orchestrator wrapper (kept import-free until actually used)
# ---------------------------------------------------------------------------

def _run_orchestrator(
    image,
    goal: str,
    max_steps: int,
    max_retries: int,
    model_path: str,
    device: str,
) -> Dict[str, Any]:
    """Run the agentic orchestrator and return a serializable result dict."""
    from florence_forge.agentic import AgenticOrchestrator, OrchestratorConfig
    from florence_forge.cli.commands_agentic import InferenceEngineAdapter
    from florence_forge.deployment.inference import InferenceEngine
    from florence_forge.utils.visualization_export import generate_step_visualization

    if image is None:
        return {"error": "No image provided"}
    if not goal or not goal.strip():
        return {"error": "Goal cannot be empty"}

    pil_img = image.convert("RGB") if hasattr(image, "convert") else image

    engine = InferenceEngine(
        model=model_path,
        device=device,
        batch_size=1,
        use_amp=False,
    )
    backend = InferenceEngineAdapter(engine)
    orch = AgenticOrchestrator(
        backend,
        OrchestratorConfig(
            max_steps=max_steps,
            max_retries=max_retries,
            emit_transcript=True,
        ),
    )

    result = orch.run(image=pil_img, goal=goal)
    result_dict = result.to_dict()

    # Attach visualizations
    for step in result_dict.get("steps", []):
        viz = generate_step_visualization(pil_img, step)
        if viz:
            step["visualization"] = viz

    return result_dict


# ---------------------------------------------------------------------------
# UI builders
# ---------------------------------------------------------------------------

def _build_step_markdown(steps: List[Dict[str, Any]]) -> str:
    """Build a markdown timeline from step records."""
    if not steps:
        return "_No steps executed._"
    lines: List[str] = ["### 步骤时间轴", ""]
    for i, step in enumerate(steps, 1):
        status = "✅" if step.get("verified") else "❌"
        tool = step.get("tool_call", "unknown")
        intent = step.get("intent", "")
        attempts = step.get("attempts", 1)
        attempts_text = f" (重试 {attempts} 次)" if attempts > 1 else ""
        raw = step.get("raw_output", "")[:120]
        issues = step.get("issues", [])
        issue_text = ""
        if issues:
            issue_text = f"\n   - ⚠️ 问题: {', '.join(issues)}"
        lines.append(
            f"{status} **Step {i}**: `{intent}` → `{tool}`{attempts_text}\n"
            f"   - 输出: `{raw}`{issue_text}"
        )
    return "\n".join(lines)


def _build_state_markdown(state: Dict[str, Any]) -> str:
    """Build a markdown summary from AgentState dict."""
    lines: List[str] = ["### AgentState 汇总", ""]
    detected = state.get("detected_objects", [])
    if detected:
        lines.append(f"- 🎯 检测到 {len(detected)} 个对象")
    text = state.get("extracted_text", [])
    if text:
        joined = " | ".join(t[:40] for t in text[:3])
        lines.append(f"- 📝 提取文字: {joined}")
    counts = state.get("counts", {})
    if counts:
        lines.append(f"- 🔢 计数: {counts}")
    regions = state.get("located_regions", [])
    if regions:
        lines.append(f"- 📍 定位区域: {len(regions)} 个")
    descriptions = state.get("descriptions", [])
    if descriptions:
        lines.append(f"- 🖼️ 描述: {descriptions[0][:80]}")
    issues = state.get("pending_issues", [])
    if issues:
        lines.append(f"- ⚠️ 未解决问题: {len(issues)} 个")
    if len(lines) == 2:
        lines.append("_暂无观察结果_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio app factory
# ---------------------------------------------------------------------------

def create_gradio_app(
    model_path: str = "microsoft/Florence-2-base",
    device: str = "auto",
) -> Any:
    """Create a Gradio Blocks app for Agentic visual reasoning.

    Args:
        model_path: Model path or Hugging Face hub ID.
        device: torch device string.

    Returns:
        A ``gradio.Blocks`` app instance.
    """
    if not GRADIO_AVAILABLE:
        raise ImportError(
            "Gradio is not installed. "
            "Install with: pip install 'florence-forge[demo]'"
        )

    with gr.Blocks(title="FlorenceForge Agentic Visual Reasoning") as demo:
        gr.Markdown(
            """
            # 🎨 FlorenceForge Agentic 多步视觉推理

            上传图像并输入目标，系统将自动分解任务、调用 Florence-2 原生工具、
            验证结果并汇总答案。

            **示例目标**:
            - "检测所有汽车并计数"
            - "读取路牌文字并描述场景"
            - "定位红色盒子并描述其内容"
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="📸 上传图像",
                    height=400,
                )
                goal_input = gr.Textbox(
                    label="🎯 目标",
                    placeholder="例如: detect all objects and count the cars",
                    value="detect all objects and count the cars",
                )
                with gr.Row():
                    max_steps = gr.Slider(
                        minimum=1, maximum=20, value=12, step=1,
                        label="最大步骤数",
                    )
                    max_retries = gr.Slider(
                        minimum=0, maximum=5, value=1, step=1,
                        label="最大重试次数",
                    )
                run_btn = gr.Button("🚀 运行 Agentic 推理", variant="primary")
                clear_btn = gr.Button("🗑️ 清空")

            with gr.Column(scale=2):
                final_answer = gr.Markdown(label="最终答案")
                with gr.Row():
                    steps_md = gr.Markdown(label="步骤时间轴")
                    state_md = gr.Markdown(label="AgentState")
                transcript_json = gr.JSON(label="完整 Transcript", visible=False)
                raw_json = gr.JSON(label="原始结果 (JSON)")

        # ------------------------------------------------------------------
        # Event handlers
        # ------------------------------------------------------------------
        def on_run(image, goal, max_steps_val, max_retries_val):
            result = _run_orchestrator(
                image, goal, int(max_steps_val), int(max_retries_val),
                model_path, device,
            )
            if "error" in result:
                return (
                    f"**错误**: {result['error']}",
                    "", "", None, result
                )
            answer = result.get("final_answer", "")
            steps = result.get("steps", [])
            state = result.get("state", {})
            transcript = result.get("transcript", "")
            return (
                f"### ✅ 最终答案\n\n{answer}",
                _build_step_markdown(steps),
                _build_state_markdown(state),
                transcript,
                result,
            )

        def on_clear():
            return (
                "", "", "", None, None,
                None, "", "", "", None,
            )

        run_btn.click(
            fn=on_run,
            inputs=[image_input, goal_input, max_steps, max_retries],
            outputs=[final_answer, steps_md, state_md, transcript_json, raw_json],
        )

        clear_btn.click(
            fn=on_clear,
            outputs=[image_input, goal_input, final_answer, steps_md, state_md,
                     transcript_json, raw_json, max_steps, max_retries],
        )

    return demo


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlorenceForge Agentic Gradio WebUI"
    )
    parser.add_argument(
        "--model", "-m", default="microsoft/Florence-2-base",
        help="模型路径或 Hugging Face ID"
    )
    parser.add_argument(
        "--device", "-d", default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="推理设备",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Gradio 服务器地址 (默认: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=7860,
        help="Gradio 服务器端口 (默认: 7860)",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="生成 Gradio 公共分享链接",
    )
    args = parser.parse_args()

    if not GRADIO_AVAILABLE:
        logger.error(
            "Gradio 未安装。请运行: pip install 'florence-forge[demo]'"
        )
        raise SystemExit(1)

    demo = create_gradio_app(model_path=args.model, device=args.device)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
