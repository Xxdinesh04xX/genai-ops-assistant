from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Any, Dict, List

from tools.registry import TOOL_REGISTRY


class ExecutorAgent:
    def __init__(self) -> None:
        self.registry = TOOL_REGISTRY

    def execute(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        steps = plan.get("steps", [])
        if len(steps) <= 1:
            return [self._run_step(step) for step in steps]

        results: List[Dict[str, Any] | None] = [None] * len(steps)
        max_workers = min(len(steps), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(self._run_step, step): idx for idx, step in enumerate(steps)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    step = steps[idx]
                    results[idx] = {
                        "step_id": step.get("id"),
                        "tool": step.get("tool"),
                        "args": step.get("args") or {},
                        "status": "error",
                        "error": str(exc),
                    }
        return [result for result in results if result is not None]

    def retry_failed(
        self,
        plan: Dict[str, Any],
        step_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        updated: List[Dict[str, Any]] = []
        step_map = {step["id"]: step for step in plan.get("steps", [])}
        for result in step_results:
            if result.get("status") == "error":
                step = step_map.get(result.get("step_id"))
                if step:
                    updated.append(self._run_step(step))
                else:
                    updated.append(result)
            else:
                updated.append(result)
        return updated

    def _run_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = step.get("tool")
        args = step.get("args") or {}
        start = time.perf_counter()
        if tool_name not in self.registry:
            return {
                "step_id": step.get("id"),
                "tool": tool_name,
                "args": args,
                "status": "error",
                "error": f"Unknown tool: {tool_name}",
                "duration_ms": int((time.perf_counter() - start) * 1000),
            }
        try:
            output = self.registry[tool_name](**args)
            return {
                "step_id": step.get("id"),
                "tool": tool_name,
                "args": args,
                "status": "success",
                "output": output,
                "duration_ms": int((time.perf_counter() - start) * 1000),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "step_id": step.get("id"),
                "tool": tool_name,
                "args": args,
                "status": "error",
                "error": str(exc),
                "duration_ms": int((time.perf_counter() - start) * 1000),
            }
