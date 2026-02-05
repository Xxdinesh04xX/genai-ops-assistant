# AI Ops Assistant

Multi-agent AI Operations Assistant with Planner → Executor → Verifier flow, LLM reasoning, and real API integrations.

## Setup (Localhost)
1. Create a virtual environment (optional but recommended).
2. Change into the project directory:
   ```bash
   cd ai_ops_assistant
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create `.env` from the example:
   ```bash
   copy .env.example .env
   ```
5. Add your `LLM_API_KEY` to `.env`.
   - Default is Groq (free tier) using `LLM_BASE_URL=https://api.groq.com/openai/v1`
     and `LLM_MODEL=llama-3.1-8b-instant`.

Run the server (one command):
```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.
UI: `http://127.0.0.1:8000/ui`
History UI: `http://127.0.0.1:8000/history`
Swagger: `http://127.0.0.1:8000/docs`

## Environment Variables
Required:
- `LLM_API_KEY`

Optional:
- `LLM_MODEL` (default: `llama-3.1-8b-instant` for Groq)
- `LLM_BASE_URL` (default: Groq base URL)
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` are supported as fallbacks
- `GITHUB_TOKEN` (increases GitHub API rate limits)

See `.env.example` for all fields.

## Architecture (Agents + Tools)
- **Planner Agent**: Uses the LLM to convert the user task into a structured JSON plan with ordered tool steps.
- **Executor Agent**: Executes steps in parallel when possible and records per-tool duration.
- **Verifier Agent**: Re-runs failed steps once, validates completeness, and produces the final structured JSON response.
- **Auto-Replan**: If completeness is partial, the system attempts one corrective replan.
- **UI**: Web UI with history, suggestions, prompt enhancement, explain mode, PDF export, and metrics.

## Integrated APIs
- GitHub Search API (`https://api.github.com/search/repositories`)
- Open-Meteo Geocoding + Weather (`https://geocoding-api.open-meteo.com`, `https://api.open-meteo.com`)

## Tools
- `github_search` — Search GitHub repositories by keyword.
- `weather_current` — Current weather by city.
- `llm_generate` — General writing tasks (e.g., speeches, summaries).
- `/suggest` — Generate similar prompts.
- `/enhance` — Rewrite prompts to be more specific.
- `/explain` — Explain outputs in simple language.

## Example Prompts
1. "Find top 3 FastAPI repositories and give current weather in Mumbai."
2. "Show popular React UI libraries and weather in Bangalore."
3. "Get trending Python repos for data science and current weather in Delhi."
4. "Search GitHub for 'vector database' and tell me the weather in Pune."
5. "List top 5 Rust web frameworks and current weather in Chennai."
6. "Find top 3 Amazon clone repositories."
7. "Write a 60-second self-introduction speech."
8. "Explain in simple words what you found and why it matters."

## Known Limitations / Tradeoffs
- Planner output depends on LLM JSON compliance; retries are used but still can fail on malformed outputs.
- Tool retries are limited to one attempt to keep latency low.
- UI history is stored locally in the browser (not persisted server-side).
- Suggestions and explanations depend on LLM quality.

## Quick Test (cURL)
```bash
curl -X POST http://127.0.0.1:8000/run ^
  -H "Content-Type: application/json" ^
  -d "{\"task\":\"Find top 3 FastAPI repositories and give current weather in Mumbai.\"}"
```

## Additional Endpoints
- `POST /suggest` — returns similar prompts
- `POST /enhance` — returns a rewritten prompt
- `POST /explain` — returns a simple explanation of the output
