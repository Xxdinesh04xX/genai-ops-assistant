import json
from typing import Any, Dict, List


PLANNER_SYSTEM = (
    "You are the Planner agent for an AI operations assistant. "
    "Convert the user's task into a step-by-step JSON plan. "
    "Only use the available tools. Output valid JSON only."
)

VERIFIER_SYSTEM = (
    "You are the Verifier agent. Validate tool outputs, detect gaps, "
    "and produce a clean, structured final response. "
    "Output valid JSON only."
)

SUGGEST_SYSTEM = (
    "You suggest helpful follow-up prompts based on the user's task. "
    "Output valid JSON only."
)

ENHANCE_SYSTEM = (
    "You rewrite user prompts to be clearer and more tool-ready. "
    "Output valid JSON only."
)

EXPLAIN_SYSTEM = (
    "You explain AI tool outputs in simple language for a non-technical user. "
    "Output valid JSON only."
)


def build_planner_user_prompt(task: str, tool_specs: List[Dict[str, Any]]) -> str:
    schema = {
        "goal": "string",
        "steps": [
            {
                "id": "string",
                "tool": "string from available tools",
                "args": "object",
                "purpose": "string",
            }
        ],
    }
    return (
        f"Task: {task}\n\n"
        f"Available tools (JSON):\n{json.dumps(tool_specs, indent=2)}\n\n"
        "Return a JSON object that matches this schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Rules:\n"
        "- Use 1-4 steps.\n"
        "- Each step must map to exactly one tool.\n"
        "- Do not repeat the same tool with identical args.\n"
        "- Prefer the minimum number of steps needed to answer the task.\n"
        "- args must match the tool's argument names.\n"
        "- Use correct JSON types (numbers as numbers, not strings).\n"
        "- If the task is a general writing request, use llm_generate.\n"
        "- Only use GitHub or Weather tools when explicitly needed.\n"
        "- Do not include any text outside JSON."
    )


def build_verifier_user_prompt(
    task: str,
    plan: Dict[str, Any],
    step_results: List[Dict[str, Any]],
) -> str:
    schema = {
        "final_answer": "string",
        "data": "object",
        "sources": ["string URLs"],
        "limitations": ["string"],
        "completeness": "complete|partial",
    }
    return (
        f"Task: {task}\n\n"
        f"Plan:\n{json.dumps(plan, indent=2)}\n\n"
        f"Step results:\n{json.dumps(step_results, indent=2)}\n\n"
        "Return a JSON object that matches this schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Guidelines:\n"
        "- Use the tool outputs as ground truth.\n"
        "- If any step failed or lacks data, set completeness to partial "
        "and add limitations.\n"
        "- Include source URLs from tool outputs when available.\n"
        "- Do not include any text outside JSON."
    )


def build_suggest_user_prompt(task: str, tool_specs: List[Dict[str, Any]]) -> str:
    schema = {"suggestions": ["string"]}
    return (
        f"Task: {task}\n\n"
        f"Available tools (JSON):\n{json.dumps(tool_specs, indent=2)}\n\n"
        "Return 3-5 short, similar prompts that can be solved using the tools.\n"
        "Return a JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "Rules:\n"
        "- Keep each suggestion under 120 characters.\n"
        "- Use only relevant tools.\n"
        "- Do not include any text outside JSON."
    )


def build_enhance_user_prompt(task: str, tool_specs: List[Dict[str, Any]]) -> str:
    schema = {"enhanced_prompt": "string"}
    return (
        f"User prompt: {task}\n\n"
        f"Available tools (JSON):\n{json.dumps(tool_specs, indent=2)}\n\n"
        "Rewrite the prompt to be clear, specific, and actionable for the tools.\n"
        "Return a JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "Rules:\n"
        "- Keep it under 200 characters.\n"
        "- Preserve user intent.\n"
        "- Do not include any text outside JSON."
    )


def build_explain_user_prompt(task: str, final: Dict[str, Any]) -> str:
    schema = {"explanation": "string"}
    return (
        f"User task: {task}\n\n"
        f"Final output:\n{json.dumps(final, indent=2)}\n\n"
        "Explain the result in 3-5 short sentences, simple English.\n"
        "Return a JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "Rules:\n"
        "- Avoid jargon.\n"
        "- Keep it under 120 words.\n"
        "- Do not include any text outside JSON."
    )
