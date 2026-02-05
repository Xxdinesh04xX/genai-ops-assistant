from typing import Any, Dict, List

from llm.client import LLMClient
from llm.prompts import VERIFIER_SYSTEM, build_verifier_user_prompt


class VerifierAgent:
    def __init__(self, executor, client: LLMClient | None = None) -> None:
        self.executor = executor
        self.client = client or LLMClient()

    def _compact_step_results(self, step_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compacted: List[Dict[str, Any]] = []
        for result in step_results:
            output = result.get("output")
            if isinstance(output, dict) and "items" in output:
                items = output.get("items", [])
                trimmed_items = []
                for item in items[:5]:
                    trimmed_items.append(
                        {
                            "name": item.get("name"),
                            "full_name": item.get("full_name"),
                            "url": item.get("url"),
                            "stars": item.get("stars"),
                            "language": item.get("language"),
                        }
                    )
                compact_output = {
                    "query": output.get("query"),
                    "total_count": output.get("total_count"),
                    "items": trimmed_items,
                    "source_url": output.get("source_url"),
                }
            else:
                compact_output = output

            compacted.append(
                {
                    "step_id": result.get("step_id"),
                    "tool": result.get("tool"),
                    "status": result.get("status"),
                    "output": compact_output,
                    "error": result.get("error"),
                }
            )
        return compacted

    def verify(
        self,
        task: str,
        plan: Dict[str, Any],
        step_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        repaired_results = self.executor.retry_failed(plan, step_results)
        missing = [r for r in repaired_results if r.get("status") != "success"]
        compact_results = self._compact_step_results(repaired_results)
        user_prompt = build_verifier_user_prompt(task, plan, compact_results)
        verification = self.client.chat_json(VERIFIER_SYSTEM, user_prompt)

        allowed_sources: List[str] = []
        repo_items: List[Dict[str, Any]] = []
        weather_info: Dict[str, Any] | None = None
        for result in repaired_results:
            output = result.get("output")
            if isinstance(output, dict):
                source_url = output.get("source_url")
                if source_url:
                    allowed_sources.append(source_url)
                if result.get("tool") == "github_search":
                    items = output.get("items", [])
                    if isinstance(items, list):
                        repo_items.extend([item for item in items if isinstance(item, dict)])
                if result.get("tool") == "weather_current":
                    weather_info = output
        for item in repo_items:
            repo_url = item.get("url")
            if repo_url:
                allowed_sources.append(repo_url)
        allowed_sources = list(dict.fromkeys(allowed_sources))

        tools_used = {
            step.get("tool")
            for step in plan.get("steps", [])
            if isinstance(step, dict) and step.get("tool")
        }
        if tools_used == {"llm_generate"}:
            speech_text = ""
            for result in repaired_results:
                if result.get("tool") == "llm_generate" and result.get("status") == "success":
                    output = result.get("output")
                    if isinstance(output, dict):
                        speech_text = str(output.get("text") or "").strip()
                        break
            if speech_text:
                word_count = len(speech_text.split())
                verification["final_answer"] = speech_text
                verification["data"] = {
                    "kind": "introduction_speech",
                    "word_count": word_count,
                }
                allowed_sources = []

        if tools_used & {"github_search", "weather_current"}:
            parts: List[str] = []
            if repo_items:
                repo_bits: List[str] = []
                for item in repo_items[:3]:
                    name = item.get("full_name") or item.get("name")
                    url = item.get("url")
                    stars = item.get("stars")
                    if name and url:
                        if stars is not None:
                            repo_bits.append(f"{name} ({url}, {stars}★)")
                        else:
                            repo_bits.append(f"{name} ({url})")
                if repo_bits:
                    parts.append("Top repositories: " + "; ".join(repo_bits) + ".")
            if isinstance(weather_info, dict):
                location = weather_info.get("location")
                summary = weather_info.get("weather_summary")
                temp = weather_info.get("temperature_c")
                wind = weather_info.get("wind_kph")
                weather_sentence = "Current weather"
                if location:
                    weather_sentence += f" in {location}"
                if summary or temp is not None:
                    details = []
                    if summary:
                        details.append(str(summary))
                    if temp is not None:
                        details.append(f"{temp}°C")
                    if wind is not None:
                        details.append(f"wind {wind} kph")
                    weather_sentence += ": " + ", ".join(details) + "."
                else:
                    weather_sentence += "."
                parts.append(weather_sentence)
            if parts:
                verification["final_answer"] = " ".join(parts).strip()

        verification["sources"] = allowed_sources

        if missing:
            if not isinstance(verification.get("limitations"), list):
                verification["limitations"] = []
            verification["limitations"].append(
                "Some tools failed after retry; partial data returned."
            )
            verification["completeness"] = "partial"

        return verification
