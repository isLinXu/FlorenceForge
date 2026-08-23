"""FastAPI routes for Agentic orchestration and visual reasoning.

Provides synchronous and streaming (SSE) endpoints that expose the
:class:`~florence_forge.agentic.AgenticOrchestrator` over HTTP.

Usage::

    from florence_forge.deployment.server import create_server
    from florence_forge.deployment.agentic_api import register_agentic_routes

    server = create_server(...)
    register_agentic_routes(server.app, server.inference_backend)
    server.run()
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, FastAPI, Form, HTTPException, UploadFile, File
    from fastapi.responses import StreamingResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = FastAPI = Form = HTTPException = UploadFile = File = None  # type: ignore
    StreamingResponse = None  # type: ignore


def _require_fastapi():
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for agentic API routes. "
            "Install with: pip install fastapi uvicorn"
        )


# ---------------------------------------------------------------------------
# Lightweight backend adapter (avoids importing CLI modules)
# ---------------------------------------------------------------------------

class _PredictTaskAdapter:
    """Wrap any inference engine/backend to expose ``predict_task``."""

    def __init__(self, engine: Any):
        self._engine = engine

    def predict_task(
        self,
        images: Any,
        task_name: str,
        text_input: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        from florence_forge.core.tasks import get_task_config_typed

        task_config = get_task_config_typed(task_name)
        result = self._engine.predict(images, task_prompt=task_config.prompt, text_input=text_input)
        if isinstance(result, str):
            return result
        return str(result)


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register_agentic_routes(app: "FastAPI", inference_backend: Any) -> None:
    """Add Agentic orchestration endpoints to a FastAPI application.

    Args:
        app: The FastAPI app instance.
        inference_backend: Any backend with a ``predict`` method that accepts
            ``task_prompt`` and ``text_input`` (e.g. :class:`InferenceEngine`
            or :class:`NativeInferenceBackend`).
    """
    _require_fastapi()
    router = APIRouter(prefix="/agentic", tags=["agentic"])

    # ------------------------------------------------------------------
    # 1. List available tools
    # ------------------------------------------------------------------
    @router.get("/tools")
    async def list_tools() -> Dict[str, Any]:
        from florence_forge.agentic import list_tools

        tools = [
            {
                "intent": t.intent,
                "task_name": t.task_name,
                "needs_text_input": t.needs_text_input,
                "output_kind": t.output_kind,
                "description": t.description,
                "keywords": t.keywords,
                "prompt": t.prompt,
            }
            for t in list_tools()
        ]
        return {"tools": tools, "count": len(tools)}

    # ------------------------------------------------------------------
    # 2. Synchronous run
    # ------------------------------------------------------------------
    @router.post("/run")
    async def run_agentic(
        image: UploadFile = File(...),  # type: ignore[arg-type]
        goal: str = Form(...),  # type: ignore[arg-type]
        max_steps: int = Form(12),  # type: ignore[arg-type]
        max_retries: int = Form(1),  # type: ignore[arg-type]
        emit_transcript: bool = Form(True),  # type: ignore[arg-type]
    ) -> Dict[str, Any]:
        from PIL import Image
        from florence_forge.agentic import AgenticOrchestrator, OrchestratorConfig
        from florence_forge.utils.visualization_export import generate_step_visualization

        try:
            contents = await image.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")  # type: ignore[arg-type]

        backend = _PredictTaskAdapter(inference_backend)
        orch = AgenticOrchestrator(
            backend,
            OrchestratorConfig(
                max_steps=max_steps,
                max_retries=max_retries,
                emit_transcript=emit_transcript,
            ),
        )

        result = orch.run(image=pil_img, goal=goal)
        result_dict = result.to_dict()

        # Attach per-step visualizations
        for step in result_dict.get("steps", []):
            viz = generate_step_visualization(pil_img, step)
            if viz:
                step["visualization"] = viz

        return result_dict

    # ------------------------------------------------------------------
    # 3. SSE streaming run
    # ------------------------------------------------------------------
    @router.post("/stream")
    async def stream_agentic(
        image: UploadFile = File(...),  # type: ignore[arg-type]
        goal: str = Form(...),  # type: ignore[arg-type]
        max_steps: int = Form(12),  # type: ignore[arg-type]
        max_retries: int = Form(1),  # type: ignore[arg-type]
        emit_transcript: bool = Form(True),  # type: ignore[arg-type]
    ) -> StreamingResponse:  # type: ignore[return-value]
        from PIL import Image
        from florence_forge.agentic import AgenticOrchestrator, OrchestratorConfig
        from florence_forge.utils.visualization_export import generate_step_visualization

        try:
            contents = await image.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")  # type: ignore[arg-type]

        backend = _PredictTaskAdapter(inference_backend)
        orch = AgenticOrchestrator(
            backend,
            OrchestratorConfig(
                max_steps=max_steps,
                max_retries=max_retries,
                emit_transcript=emit_transcript,
            ),
        )

        async def event_generator():
            # Emit plan event
            plan = orch.decompose(goal)
            plan_payload = {
                "type": "plan",
                "goal": goal,
                "sub_tasks": [
                    {"index": s.index, "intent": s.intent, "goal": s.goal}
                    for s in plan.sub_tasks
                ],
                "rationale": plan.rationale,
            }
            yield f"data: {json.dumps(plan_payload, ensure_ascii=False)}\n\n"

            from florence_forge.agentic import AgentState

            agent_state = AgentState()
            steps: List[Dict[str, Any]] = []
            for sub in plan.sub_tasks:
                if len(steps) >= max_steps:
                    break
                record = orch._execute_sub_task(pil_img, sub, agent_state)
                step_dict = {
                    "sub_task_index": record.sub_task_index,
                    "intent": record.intent,
                    "tool_call": record.tool_call.describe(),
                    "raw_output": record.raw_output[:500],
                    "verified": record.verified,
                    "attempts": record.attempts,
                    "issues": record.issues,
                }
                viz = generate_step_visualization(pil_img, step_dict)
                if viz:
                    step_dict["visualization"] = viz
                steps.append(step_dict)
                yield f"data: {json.dumps({'type': 'step', 'step': step_dict}, ensure_ascii=False)}\n\n"

            final_answer = orch._aggregate(goal, agent_state, steps)
            yield f"data: {json.dumps({'type': 'done', 'final_answer': final_answer, 'state': agent_state.summarize()}, ensure_ascii=False)}\n\n"

        return StreamingResponse(  # type: ignore[return-value]
            event_generator(),
            media_type="text/event-stream",
        )

    app.include_router(router)
    logger.info("Agentic routes registered at /agentic/*")
