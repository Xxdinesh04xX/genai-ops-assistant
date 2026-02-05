from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agents.executor import ExecutorAgent
from agents.planner import PlannerAgent
from agents.verifier import VerifierAgent
from llm.client import LLMClient
from llm.prompts import (
    ENHANCE_SYSTEM,
    EXPLAIN_SYSTEM,
    SUGGEST_SYSTEM,
    build_enhance_user_prompt,
    build_explain_user_prompt,
    build_suggest_user_prompt,
)
from tools.registry import TOOL_SPECS

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

app = FastAPI(title="AI Ops Assistant", version="1.0.0")

planner = PlannerAgent()
executor = ExecutorAgent()
verifier = VerifierAgent(executor=executor)
llm_client = LLMClient()


def _extract_step_errors(step_results: list[dict]) -> list[str]:
    errors: list[str] = []
    for step in step_results:
        if step.get("status") == "error":
            tool = step.get("tool") or "unknown_tool"
            err = step.get("error") or "unknown error"
            errors.append(f"{tool}: {err}")
    return errors


def _build_replan_task(task: str, steps: list[dict], final: dict) -> str | None:
    errors = _extract_step_errors(steps)
    limitations = final.get("limitations", []) if isinstance(final, dict) else []
    if not errors and not limitations:
        return None
    parts: list[str] = []
    if errors:
        parts.append("Errors: " + "; ".join(errors))
    if limitations:
        parts.append("Limitations: " + "; ".join(map(str, limitations)))
    return (
        f"{task}\n\n"
        "Fix missing or failed info from the previous attempt. "
        "Use only the necessary tools. "
        + " ".join(parts)
    )


class TaskRequest(BaseModel):
    task: str = Field(..., min_length=3, description="Natural language task")


class SuggestRequest(BaseModel):
    task: str = Field(..., min_length=3, description="Natural language task")


class ExplainRequest(BaseModel):
    task: str = Field(..., min_length=3, description="Natural language task")
    final: dict = Field(..., description="Final output object")


@app.get("/")
def root() -> dict:
    return {"status": "ok", "message": "AI Ops Assistant is running"}


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AI Ops Assistant</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 0; background: #0b1220; color: #e6e6e6; }
      .container { max-width: 900px; margin: 40px auto; padding: 24px; background: #121a2b; border-radius: 12px; position: relative; }
      h1 { margin-top: 0; font-size: 24px; }
      textarea { width: 100%; min-height: 110px; background: #0b1220; color: #e6e6e6; border: 1px solid #2b3a5c; border-radius: 8px; padding: 12px; }
      button { margin-top: 12px; padding: 10px 16px; border: 0; border-radius: 8px; background: #4f7cff; color: white; cursor: pointer; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      pre { white-space: pre-wrap; background: #0b1220; border: 1px solid #2b3a5c; padding: 12px; border-radius: 8px; }
      .row { display: flex; gap: 16px; align-items: center; }
      .muted { color: #9fb0d1; font-size: 12px; }
      .footer { font-size: 12px; color: #9fb0d1; }
      .history-btn { position: absolute; top: 18px; right: 18px; margin-top: 0; background: #27345a; }
      .final-answer { margin-top: 12px; padding: 12px; border: 1px solid #2b3a5c; border-radius: 8px; background: #0b1220; }
      .final-title { margin: 0 0 6px; font-size: 14px; color: #9fb0d1; }
      .final-answer ul { margin: 6px 0 0 18px; padding: 0; }
      .final-answer li { margin-bottom: 6px; }
      .final-answer a { color: #8ab4ff; text-decoration: none; }
      .final-section { margin-top: 8px; }
      .final-label { font-size: 12px; color: #9fb0d1; margin-bottom: 4px; }
      .metrics { margin-top: 12px; padding: 12px; border: 1px solid #2b3a5c; border-radius: 8px; background: #0b1220; }
      .metrics-title { margin: 0 0 6px; font-size: 14px; color: #9fb0d1; }
      .metrics-row { display: flex; gap: 16px; flex-wrap: wrap; }
      .metrics-item { color: #e6e6e6; font-size: 13px; }
      .suggestions { margin-top: 12px; padding: 12px; border: 1px solid #2b3a5c; border-radius: 8px; background: #0b1220; }
      .suggestions-title { margin: 0 0 6px; font-size: 14px; color: #9fb0d1; }
      .suggestions ul { margin: 6px 0 0 18px; padding: 0; }
      .suggestions li { margin-bottom: 6px; }
      .suggestion-item { display: flex; align-items: center; gap: 8px; justify-content: space-between; }
      .suggestion-text { flex: 1; }
      .action-row { display: flex; gap: 8px; align-items: center; }
      .footer-row { margin-top: 16px; display: flex; align-items: center; justify-content: space-between; gap: 12px; }
      .footer { margin-top: 8px; }
      .modal { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; background: rgba(5, 8, 15, 0.7); z-index: 20; }
      .modal-content { width: min(700px, 92vw); max-height: 70vh; overflow: auto; background: #0b1220; border: 1px solid #2b3a5c; border-radius: 12px; padding: 16px; }
      .modal-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
      .close-btn { margin-top: 0; padding: 8px 12px; border: 0; border-radius: 8px; background: #27345a; color: white; cursor: pointer; }
      .secondary-btn { background: #3a4f7a; }
      .download-btn { margin: 0; background: #2f5cff; display: none; }
    </style>
  </head>
  <body>
    <div class="container" id="mainContainer">
      <h1>AI Ops Assistant</h1>
      <button id="historyBtn" class="history-btn">History</button>
      <div class="row">
        <div class="muted">Planner → Executor → Verifier</div>
      </div>
      <p>Enter a task and run the agents:</p>
      <textarea id="taskInput">Find top 3 FastAPI repositories and give current weather in Mumbai.</textarea>
      <br />
      <div class="action-row">
        <button id="runBtn">Run</button>
        <button id="enhanceBtn" class="secondary-btn">Enhance Prompt</button>
      </div>
      <h3>Response</h3>
      <pre id="output">{}</pre>
      <div class="final-answer">
        <div class="final-title">Final Answer</div>
        <div id="finalAnswerContent">-</div>
      </div>
      <div class="metrics">
        <div class="metrics-title">Performance</div>
        <div class="metrics-row">
          <div id="metricToolCount" class="metrics-item">Tools: -</div>
          <div id="metricToolTime" class="metrics-item">Tool time: -</div>
        </div>
      </div>
      <div class="footer-row">
        <button id="explainBtn" class="secondary-btn">Explain Output</button>
        <button id="downloadPdfBtn" class="download-btn">Download PDF</button>
      </div>
      <div class="footer">Try /docs for Swagger API explorer.</div>
      <div class="suggestions">
        <div class="suggestions-title">Suggestions</div>
        <div id="suggestionsMeta" class="muted"></div>
        <ul id="suggestionsList"></ul>
      </div>
    </div>
    <div id="explainModal" class="modal">
      <div class="modal-content">
        <div class="modal-header">
          <strong>Explanation</strong>
          <button id="closeExplain" class="close-btn">Close</button>
        </div>
        <div id="explainText"></div>
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js"></script>
    <script>
      const storageKey = "ai_ops_history_v1";
      const selectedKey = "ai_ops_selected_index";
      const btn = document.getElementById("runBtn");
      const enhanceBtn = document.getElementById("enhanceBtn");
      const explainBtn = document.getElementById("explainBtn");
      const output = document.getElementById("output");
      const historyBtn = document.getElementById("historyBtn");
      const finalAnswerContent = document.getElementById("finalAnswerContent");
      const downloadPdfBtn = document.getElementById("downloadPdfBtn");
      const suggestionsList = document.getElementById("suggestionsList");
      const suggestionsMeta = document.getElementById("suggestionsMeta");
      const metricToolCount = document.getElementById("metricToolCount");
      const metricToolTime = document.getElementById("metricToolTime");
      const explainModal = document.getElementById("explainModal");
      const closeExplain = document.getElementById("closeExplain");
      const explainText = document.getElementById("explainText");
      let lastResponse = null;

      function loadHistory() {
        try {
          return JSON.parse(localStorage.getItem(storageKey) || "[]");
        } catch (_) {
          return [];
        }
      }

      function saveHistory(entries) {
        localStorage.setItem(storageKey, JSON.stringify(entries.slice(0, 20)));
      }

      function renderFinalAnswer(data) {
        while (finalAnswerContent.firstChild) {
          finalAnswerContent.removeChild(finalAnswerContent.firstChild);
        }
        if (!data || !data.final) {
          finalAnswerContent.textContent = "-";
          return;
        }

        const finalText = data.final.final_answer || "";
        const dataBlock = data.final.data || {};

        const repoArray = Object.values(dataBlock).find(
          (value) =>
            Array.isArray(value) &&
            value.length &&
            typeof value[0] === "object" &&
            (value[0].url || value[0].full_name || value[0].name)
        );

        let weatherObj = dataBlock.current_weather || dataBlock.weather;
        if (!weatherObj) {
          weatherObj = Object.values(dataBlock).find(
            (value) =>
              value &&
              typeof value === "object" &&
              ("weather_summary" in value || "temperature" in value || "temperature_c" in value)
          );
        }

        if (repoArray) {
          const section = document.createElement("div");
          section.className = "final-section";
          const label = document.createElement("div");
          label.className = "final-label";
          label.textContent = "Repositories";
          section.appendChild(label);

          const list = document.createElement("ul");
          repoArray.slice(0, 5).forEach((repo) => {
            const li = document.createElement("li");
            const name = repo.full_name || repo.name || "Repository";
            const stars = repo.stars != null ? ` (${repo.stars}★)` : "";
            if (repo.url) {
              const link = document.createElement("a");
              link.href = repo.url;
              link.target = "_blank";
              link.rel = "noopener noreferrer";
              link.textContent = `${name}${stars}`;
              li.appendChild(link);
            } else {
              li.textContent = `${name}${stars}`;
            }
            list.appendChild(li);
          });
          section.appendChild(list);
          finalAnswerContent.appendChild(section);
        }

        if (weatherObj && typeof weatherObj === "object") {
          const section = document.createElement("div");
          section.className = "final-section";
          const label = document.createElement("div");
          label.className = "final-label";
          label.textContent = "Weather";
          section.appendChild(label);

          const parts = [];
          if (weatherObj.location) parts.push(weatherObj.location);
          if (weatherObj.weather_summary) parts.push(weatherObj.weather_summary);
          const temp =
            typeof weatherObj.temperature_c !== "undefined"
              ? weatherObj.temperature_c
              : weatherObj.temperature;
          if (temp != null) parts.push(`${temp}°C`);
          if (weatherObj.wind_kph != null) parts.push(`wind ${weatherObj.wind_kph} kph`);
          const weatherLine = document.createElement("div");
          weatherLine.textContent = parts.length ? parts.join(" • ") : "Weather details unavailable.";
          section.appendChild(weatherLine);
          finalAnswerContent.appendChild(section);
        }

        if (!repoArray && !weatherObj) {
          const paragraph = document.createElement("div");
          paragraph.textContent = finalText || "-";
          finalAnswerContent.appendChild(paragraph);
        }
      }

      function setDownloadVisibility() {
        downloadPdfBtn.style.display = lastResponse ? "block" : "none";
      }

      function renderSuggestions(data) {
        while (suggestionsList.firstChild) {
          suggestionsList.removeChild(suggestionsList.firstChild);
        }
        suggestionsMeta.textContent = "";
        if (data && data.auto_replan && data.auto_replan.triggered) {
          suggestionsMeta.textContent = "Auto-replan was triggered to fix missing data.";
        } else if (data && data.final && data.final.completeness === "partial") {
          suggestionsMeta.textContent = "Results are partial. Try a more specific prompt.";
        }
        const li = document.createElement("li");
        li.textContent = "Fetching suggestions...";
        suggestionsList.appendChild(li);
      }

      function renderMetrics(data) {
        if (!data || !data.metrics) {
          metricToolCount.textContent = "Tools: -";
          metricToolTime.textContent = "Tool time: -";
          return;
        }
        const toolCount = data.metrics.tool_count;
        const toolTime = data.metrics.tool_execution_ms;
        metricToolCount.textContent =
          typeof toolCount === "number" ? `Tools: ${toolCount}` : "Tools: -";
        if (typeof toolTime === "number") {
          const seconds = (toolTime / 1000).toFixed(2);
          metricToolTime.textContent = `Tool time: ${toolTime} ms (${seconds}s)`;
        } else {
          metricToolTime.textContent = "Tool time: -";
        }
      }

      function renderSuggestionPrompts(prompts) {
        while (suggestionsList.firstChild) {
          suggestionsList.removeChild(suggestionsList.firstChild);
        }
        if (!prompts || !prompts.length) {
          const li = document.createElement("li");
          li.textContent = "No suggestions available.";
          suggestionsList.appendChild(li);
          return;
        }
        prompts.forEach((text) => {
          const li = document.createElement("li");
          li.className = "suggestion-item";
          const span = document.createElement("span");
          span.className = "suggestion-text";
          span.textContent = text;
          const useBtn = document.createElement("button");
          useBtn.className = "secondary-btn";
          useBtn.textContent = "Use";
          useBtn.addEventListener("click", () => {
            document.getElementById("taskInput").value = text;
            document.getElementById("taskInput").scrollIntoView({ behavior: "smooth" });
          });
          li.appendChild(span);
          li.appendChild(useBtn);
          suggestionsList.appendChild(li);
        });
      }

      async function fetchSuggestions(task) {
        try {
          const res = await fetch("/suggest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task: task })
          });
          const data = await res.json();
          return Array.isArray(data.suggestions) ? data.suggestions : [];
        } catch (_) {
          return [];
        }
      }

      function sanitizePdfText(text) {
        return String(text || "")
          .replace(/°/g, " deg ")
          .replace(/[^\x20-\x7E]/g, "");
      }

      function addTextBlock(doc, text, startY) {
        const marginX = 12;
        const pageWidth = doc.internal.pageSize.getWidth();
        const pageHeight = doc.internal.pageSize.getHeight();
        const maxWidth = pageWidth - marginX * 2;
        const lines = doc.splitTextToSize(sanitizePdfText(text), maxWidth);
        let y = startY;
        lines.forEach((line) => {
          if (y > pageHeight - 12) {
            doc.addPage();
            y = 12;
          }
          doc.text(line, marginX, y);
          y += 6;
        });
        return y;
      }

      function addLinkLine(doc, text, url, startY) {
        const marginX = 14;
        const pageHeight = doc.internal.pageSize.getHeight();
        let y = startY;
        if (y > pageHeight - 12) {
          doc.addPage();
          y = 12;
        }
        doc.setTextColor(20, 63, 153);
        doc.textWithLink(sanitizePdfText(text), marginX, y, { url: url });
        doc.setTextColor(0, 0, 0);
        return y + 6;
      }

      function extractRepoArray(data) {
        const dataBlock = data && data.final && data.final.data ? data.final.data : {};
        const direct = dataBlock.repositories || dataBlock.top_fastapi_repos;
        if (Array.isArray(direct) && direct.length) return direct;
        const values = Object.values(dataBlock);
        for (let i = 0; i < values.length; i += 1) {
          const value = values[i];
          if (
            Array.isArray(value) &&
            value.length &&
            typeof value[0] === "object" &&
            (value[0].url || value[0].full_name || value[0].name)
          ) {
            return value;
          }
        }
        return null;
      }

      function extractWeather(data) {
        const dataBlock = data && data.final && data.final.data ? data.final.data : {};
        if (dataBlock.current_weather) return dataBlock.current_weather;
        if (dataBlock.weather) return dataBlock.weather;
        const values = Object.values(dataBlock);
        for (let i = 0; i < values.length; i += 1) {
          const value = values[i];
          if (
            value &&
            typeof value === "object" &&
            ("weather_summary" in value || "temperature" in value || "temperature_c" in value)
          ) {
            return value;
          }
        }
        return null;
      }

      

      historyBtn.addEventListener("click", () => {
        window.location.href = "/history";
      });

      btn.addEventListener("click", async () => {
        btn.disabled = true;
        output.textContent = "Running...";
        try {
          const res = await fetch("/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task: document.getElementById("taskInput").value })
          });
          const data = await res.json();
          output.textContent = JSON.stringify(data, null, 2);
          renderFinalAnswer(data);
          renderSuggestions(data);
          renderMetrics(data);
          lastResponse = data;
          setDownloadVisibility();
          const suggestions = await fetchSuggestions(document.getElementById("taskInput").value);
          renderSuggestionPrompts(suggestions);
          const entries = loadHistory();
          entries.unshift({
            task: document.getElementById("taskInput").value,
            response: data,
            time: new Date().toLocaleString()
          });
          saveHistory(entries);
        } catch (err) {
          output.textContent = "Error: " + err;
          lastResponse = null;
          setDownloadVisibility();
        } finally {
          btn.disabled = false;
        }
      });

      const selectedIndex = localStorage.getItem(selectedKey);
      if (selectedIndex !== null) {
        const entries = loadHistory();
        const entry = entries[Number(selectedIndex)];
        if (entry && entry.response) {
          output.textContent = JSON.stringify(entry.response, null, 2);
          document.getElementById("taskInput").value = entry.task || "";
          renderFinalAnswer(entry.response);
          renderSuggestions(entry.response);
          renderMetrics(entry.response);
          lastResponse = entry.response;
          setDownloadVisibility();
          fetchSuggestions(entry.task || "").then(renderSuggestionPrompts);
        }
        localStorage.removeItem(selectedKey);
      }

      enhanceBtn.addEventListener("click", async () => {
        const current = document.getElementById("taskInput").value;
        if (!current) return;
        enhanceBtn.disabled = true;
        enhanceBtn.textContent = "Enhancing...";
        try {
          const res = await fetch("/enhance", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task: current })
          });
          const data = await res.json();
          if (data && data.enhanced_prompt) {
            document.getElementById("taskInput").value = data.enhanced_prompt;
          }
        } catch (_) {
          // ignore
        } finally {
          enhanceBtn.disabled = false;
          enhanceBtn.textContent = "Enhance Prompt";
        }
      });

      explainBtn.addEventListener("click", async () => {
        if (!lastResponse || !lastResponse.final) {
          explainText.textContent = "Run a task first to see an explanation.";
          explainModal.style.display = "flex";
          return;
        }
        explainBtn.disabled = true;
        explainBtn.textContent = "Explaining...";
        try {
          const res = await fetch("/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task: lastResponse.task || "", final: lastResponse.final })
          });
          const data = await res.json();
          explainText.textContent = data.explanation || "No explanation available.";
          explainModal.style.display = "flex";
        } catch (_) {
          explainText.textContent = "Unable to generate explanation.";
          explainModal.style.display = "flex";
        } finally {
          explainBtn.disabled = false;
          explainBtn.textContent = "Explain Output";
        }
      });

      closeExplain.addEventListener("click", () => {
        explainModal.style.display = "none";
      });

      explainModal.addEventListener("click", (event) => {
        if (event.target === explainModal) {
          explainModal.style.display = "none";
        }
      });

      downloadPdfBtn.addEventListener("click", () => {
        if (!lastResponse || !window.jspdf) return;
        const doc = new window.jspdf.jsPDF({ unit: "mm", format: "a4" });
        doc.setFontSize(14);
        doc.text("AI Ops Assistant Response", 12, 14);
        doc.setFontSize(10);
        const taskText = lastResponse.task ? `Question: ${lastResponse.task}` : "Question: -";
        let y = 24;
        y = addTextBlock(doc, taskText, y);
        y = addTextBlock(doc, "Final Answer:", y + 4);

        const repos = extractRepoArray(lastResponse);
        const weather = extractWeather(lastResponse);

        if (repos && repos.length) {
          y = addTextBlock(doc, "Repositories:", y + 2);
          const repoCount = Math.min(3, repos.length);
          for (let i = 0; i < repoCount; i += 1) {
            const repo = repos[i];
            const name = repo.full_name || repo.name || "Repository";
            const stars =
              typeof repo.stars !== "undefined" ? ` (${repo.stars} stars)` : "";
            y = addTextBlock(doc, `${i + 1}) ${name}${stars}`, y + 2);
            if (repo.url) {
              y = addLinkLine(doc, repo.url, repo.url, y + 1);
            }
          }
        }

        if (weather && typeof weather === "object") {
          const summary = weather.weather_summary || "";
          const temp =
            typeof weather.temperature_c !== "undefined"
              ? weather.temperature_c
              : weather.temperature;
          const wind = weather.wind_kph;
          const parts = [];
          if (weather.location) parts.push(weather.location);
          if (summary) parts.push(summary);
          if (typeof temp !== "undefined") parts.push(`${temp} deg C`);
          if (typeof wind !== "undefined") parts.push(`wind ${wind} kph`);
          const weatherLine = parts.length ? parts.join(", ") : "Weather details unavailable.";
          y = addTextBlock(doc, "Weather:", y + 2);
          addTextBlock(doc, weatherLine, y + 2);
        }

        if (!repos && !weather) {
          const finalText =
            lastResponse.final && lastResponse.final.final_answer
              ? lastResponse.final.final_answer
              : "-";
          addTextBlock(doc, finalText, y + 2);
        }
        doc.save("ai_ops_response.pdf");
      });
    </script>
  </body>
</html>
"""


@app.get("/history", response_class=HTMLResponse)
def history() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AI Ops Assistant - History</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 0; background: #0b1220; color: #e6e6e6; }
      .header { display: flex; align-items: center; gap: 12px; padding: 16px 24px; border-bottom: 1px solid #2b3a5c; background: #121a2b; position: sticky; top: 0; }
      .title { margin: 0; font-size: 18px; }
      .back-btn { margin-top: 0; padding: 10px 16px; border: 0; border-radius: 8px; background: #27345a; color: white; cursor: pointer; }
      .history-list { max-width: 900px; margin: 16px auto 40px; padding: 0 24px; }
      .history-item { background: #121a2b; border: 1px solid #223252; border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; }
      .history-item:last-child { margin-bottom: 0; }
      .history-actions { display: flex; gap: 8px; margin-top: 8px; }
      .secondary-btn { padding: 8px 12px; border: 0; border-radius: 8px; background: #3a4f7a; color: white; cursor: pointer; }
      .muted { color: #9fb0d1; font-size: 12px; }
      .modal { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; background: rgba(5, 8, 15, 0.7); z-index: 20; }
      .modal-content { width: min(900px, 92vw); max-height: 82vh; overflow: auto; background: #0b1220; border: 1px solid #2b3a5c; border-radius: 12px; padding: 16px; }
      .modal-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
      .close-btn { margin-top: 0; padding: 8px 12px; border: 0; border-radius: 8px; background: #27345a; color: white; cursor: pointer; }
      pre { white-space: pre-wrap; background: #0b1220; border: 1px solid #2b3a5c; padding: 12px; border-radius: 8px; }
    </style>
  </head>
  <body>
    <div class="header">
      <button id="backBtn" class="back-btn">Back</button>
      <h2 class="title">Past prompts</h2>
      <button id="downloadAllBtn" class="back-btn" style="margin-left:auto;">Download All PDF</button>
    </div>
    <div id="historyList" class="history-list"></div>
    <div id="resultModal" class="modal">
      <div class="modal-content">
        <div class="modal-header">
          <strong>Result</strong>
          <button id="closeModal" class="close-btn">Close</button>
        </div>
        <pre id="resultOutput">{}</pre>
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js"></script>
    <script>
      const storageKey = "ai_ops_history_v1";
      const historyList = document.getElementById("historyList");
      const backBtn = document.getElementById("backBtn");
      const downloadAllBtn = document.getElementById("downloadAllBtn");
      const resultModal = document.getElementById("resultModal");
      const closeModal = document.getElementById("closeModal");
      const resultOutput = document.getElementById("resultOutput");

      function loadHistory() {
        try {
          return JSON.parse(localStorage.getItem(storageKey) || "[]");
        } catch (_) {
          return [];
        }
      }

      function renderHistory() {
        const entries = loadHistory();
        if (!entries.length) {
          historyList.innerHTML = "<div class=\\"muted\\">No history yet.</div>";
          return;
        }
        historyList.innerHTML = entries
          .map((entry, idx) => {
            const safeTask = (entry.task || "").replace(/</g, "&lt;");
            return `
              <div class="history-item">
                <div><strong>Prompt:</strong> ${safeTask}</div>
                <div class="muted">${entry.time || ""}</div>
                <div class="history-actions">
                  <button class="secondary-btn" data-index="${idx}">Show Result</button>
                </div>
              </div>
            `;
          })
          .join("");

        historyList.querySelectorAll("button[data-index]").forEach((btnEl) => {
          btnEl.addEventListener("click", (event) => {
            const idx = Number(event.currentTarget.getAttribute("data-index"));
            const entries = loadHistory();
            const entry = entries[idx];
            if (entry && entry.response) {
              resultOutput.textContent = JSON.stringify(entry.response, null, 2);
              resultModal.style.display = "flex";
            }
          });
        });
      }

      backBtn.addEventListener("click", () => {
        window.location.href = "/ui";
      });

      function sanitizePdfText(text) {
        return String(text || "")
          .replace(/°/g, " deg ")
          .replace(/[^\x20-\x7E]/g, "");
      }

      function addTextBlock(doc, text, startY) {
        const marginX = 12;
        const pageWidth = doc.internal.pageSize.getWidth();
        const pageHeight = doc.internal.pageSize.getHeight();
        const maxWidth = pageWidth - marginX * 2;
        const lines = doc.splitTextToSize(sanitizePdfText(text), maxWidth);
        let y = startY;
        lines.forEach((line) => {
          if (y > pageHeight - 12) {
            doc.addPage();
            y = 12;
          }
          doc.text(line, marginX, y);
          y += 6;
        });
        return y;
      }

      downloadAllBtn.addEventListener("click", () => {
        if (!window.jspdf) return;
        const entries = loadHistory();
        if (!entries.length) return;
        const doc = new window.jspdf.jsPDF({ unit: "mm", format: "a4" });
        doc.setFontSize(14);
        doc.text("AI Ops Assistant - History", 12, 14);
        doc.setFontSize(10);
        let y = 24;
        entries.forEach((entry) => {
          const taskText = entry.task ? `Question: ${entry.task}` : "Question: -";
          y = addTextBlock(doc, taskText, y);
          const finalAnswer =
            entry.response &&
            entry.response.final &&
            entry.response.final.final_answer
              ? entry.response.final.final_answer
              : "-";
          y = addTextBlock(doc, `Final Answer: ${finalAnswer}`, y + 2);
          y += 6;
          if (y > doc.internal.pageSize.getHeight() - 20) {
            doc.addPage();
            y = 12;
          }
        });
        doc.save("ai_ops_history.pdf");
      });

      closeModal.addEventListener("click", () => {
        resultModal.style.display = "none";
      });

      resultModal.addEventListener("click", (event) => {
        if (event.target === resultModal) {
          resultModal.style.display = "none";
        }
      });

      renderHistory();
    </script>
  </body>
</html>
"""


@app.post("/run")
def run_task(request: TaskRequest) -> dict:
    try:
        plan = planner.create_plan(request.task)
        steps = executor.execute(plan)
        final = verifier.verify(request.task, plan, steps)
        tool_time_ms = sum(
            int(step.get("duration_ms", 0))
            for step in steps
            if isinstance(step, dict)
        )
        auto_replan = None
        if final.get("completeness") == "partial":
            replan_task = _build_replan_task(request.task, steps, final)
            if replan_task:
                replan_plan = planner.create_plan(replan_task)
                replan_steps = executor.execute(replan_plan)
                replan_final = verifier.verify(request.task, replan_plan, replan_steps)
                auto_replan = {
                    "triggered": True,
                    "reason": "Previous attempt incomplete",
                    "task": replan_task,
                    "plan": replan_plan,
                    "steps": replan_steps,
                    "final": replan_final,
                }
                if replan_final.get("completeness") == "complete":
                    plan = replan_plan
                    steps = replan_steps
                    final = replan_final
        return {
            "task": request.task,
            "plan": plan,
            "steps": steps,
            "final": final,
            "auto_replan": auto_replan,
            "metrics": {
                "tool_execution_ms": tool_time_ms,
                "tool_count": len(steps),
            },
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/suggest")
def suggest(request: SuggestRequest) -> dict:
    try:
        user_prompt = build_suggest_user_prompt(request.task, TOOL_SPECS)
        data = llm_client.chat_json(SUGGEST_SYSTEM, user_prompt)
        suggestions = data.get("suggestions")
        if not isinstance(suggestions, list):
            suggestions = []
        cleaned = [str(s).strip() for s in suggestions if str(s).strip()]
        return {"suggestions": cleaned[:5]}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/enhance")
def enhance(request: SuggestRequest) -> dict:
    try:
        user_prompt = build_enhance_user_prompt(request.task, TOOL_SPECS)
        data = llm_client.chat_json(ENHANCE_SYSTEM, user_prompt)
        enhanced = data.get("enhanced_prompt", "")
        return {"enhanced_prompt": str(enhanced).strip()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/explain")
def explain(request: ExplainRequest) -> dict:
    try:
        user_prompt = build_explain_user_prompt(request.task, request.final)
        data = llm_client.chat_json(EXPLAIN_SYSTEM, user_prompt)
        explanation = data.get("explanation", "")
        return {"explanation": str(explanation).strip()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
