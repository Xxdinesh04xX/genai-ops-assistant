from typing import Any, Dict, List

from llm.client import LLMClient
from llm.prompts import PLANNER_SYSTEM, build_planner_user_prompt
from tools.registry import TOOL_REGISTRY, TOOL_SPECS


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def create_plan(self, task: str) -> Dict[str, Any]:
        user_prompt = build_planner_user_prompt(task, TOOL_SPECS)
        plan = self.client.chat_json(PLANNER_SYSTEM, user_prompt)
        return self._normalize_plan(plan, task)

    def _normalize_plan(self, plan: Dict[str, Any], task: str) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            raise ValueError("Planner output is not a JSON object")
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("Planner output missing steps")

        normalized_steps: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps, start=1):
            tool = step.get("tool")
            if not tool or tool not in TOOL_REGISTRY:
                raise ValueError(f"Unknown tool in plan: {tool}")
            normalized_steps.append(
                {
                    "id": step.get("id") or f"step_{idx}",
                    "tool": tool,
                    "args": step.get("args") or {},
                    "purpose": step.get("purpose") or "",
                }
            )

        return {"goal": plan.get("goal") or task, "steps": normalized_steps}
