"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { createRun, getRun, RunDetails } from "@/lib/api";

type ActiveTab = "requirements" | "code" | "tests" | "review";

type ModelOption = {
  value: string;
  label: string;
  tier: "light" | "medium" | "heavy";
  note: string;
};

const MODEL_OPTIONS: ModelOption[] = [
  { value: "tinyllama",      label: "TinyLlama 1.1B",  tier: "light",  note: "~600 MB — fastest, good for dev" },
  { value: "phi3:mini",      label: "Phi-3 Mini 3.8B", tier: "light",  note: "~2.2 GB — smart & fast" },
  { value: "qwen2:0.5b",     label: "Qwen2 0.5B",      tier: "light",  note: "~350 MB — ultra-light" },
  { value: "gemma:2b",       label: "Gemma 2B",        tier: "light",  note: "~1.4 GB — lightweight" },
  { value: "mistral:7b",     label: "Mistral 7B",      tier: "medium", note: "~4.1 GB — balanced" },
  { value: "qwen:7b",        label: "Qwen 7B",         tier: "medium", note: "~4.5 GB — balanced" },
  { value: "llama3:8b",      label: "Llama 3 8B",      tier: "heavy",  note: "~4.7 GB — recommended" },
  { value: "codellama:7b",   label: "Code Llama 7B",   tier: "heavy",  note: "~3.8 GB — code-focused" },
  { value: "__custom__",     label: "Custom model...", tier: "light",  note: "Enter a model name manually" },
];

const TIER_LABEL: Record<ModelOption["tier"], string> = {
  light: "light",
  medium: "medium",
  heavy: "heavy",
};

const defaultPrompt =
  "Build a password reset module with secure token generation and expiry validation.";

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

export function Dashboard() {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [selectedModel, setSelectedModel] = useState("phi3:mini");
  const [customModel, setCustomModel] = useState("");
  const [ollamaUrl, setOllamaUrl] = useState("http://localhost:11434");
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<RunDetails | null>(null);
  const [tab, setTab] = useState<ActiveTab>("requirements");
  const [submitError, setSubmitError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const resolvedModel =
    selectedModel === "__custom__" ? customModel.trim() : selectedModel;

  const selectedOption = MODEL_OPTIONS.find((m) => m.value === selectedModel);

  useEffect(() => {
    if (!runId) return;
    let active = true;

    const poll = async () => {
      try {
        const details = await getRun(runId);
        if (active) setRun(details);
      } catch (err) {
        if (active) setSubmitError((err as Error).message);
      }
    };

    poll();
    const id = setInterval(poll, 1500);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [runId]);

  const isActive =
    isSubmitting ||
    run?.status === "queued" ||
    run?.status === "running";

  const statusClass = useMemo(() => {
    if (!run) return "";
    return `status-${run.status}`;
  }, [run]);

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSubmitError("");
    setIsSubmitting(true);
    setRun(null);

    try {
      const summary = await createRun({
        user_prompt: prompt.trim(),
        model_name: resolvedModel,
        ollama_base_url: ollamaUrl.trim(),
      });
      setRunId(summary.run_id);
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="shell">
      <header className="site-header">
        <p className="overline">MAS SDLC — Multi-Agent Pipeline</p>
        <h1>AI Software<br />Delivery Team</h1>
        <p>
          Four specialised agents work sequentially to turn a plain-English
          feature request into requirements, code, tests, and a review —
          all running locally via Ollama.
        </p>
      </header>

      <div className="workspace">
        {/* ── Left: run form ── */}
        <section className="panel">
          <p className="panel-title">New Pipeline Run</p>

          <form onSubmit={onSubmit}>
            <div className="field">
              <label htmlFor="prompt">Feature Request</label>
              <textarea
                id="prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                minLength={5}
                required
                rows={6}
                placeholder="Describe the feature you want to build..."
              />
            </div>

            <div className="field-row">
              <div className="field">
                <label htmlFor="model">Model</label>
                <select
                  id="model"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  {MODEL_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
                {selectedOption && selectedOption.value !== "__custom__" && (
                  <span style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 6 }}>
                    <span className={`model-badge ${selectedOption.tier}`}>
                      {TIER_LABEL[selectedOption.tier]}
                    </span>
                    {" "}{selectedOption.note}
                  </span>
                )}
              </div>

              <div className="field">
                <label htmlFor="ollama-url">Ollama URL</label>
                <input
                  id="ollama-url"
                  value={ollamaUrl}
                  onChange={(e) => setOllamaUrl(e.target.value)}
                  required
                />
              </div>
            </div>

            {selectedModel === "__custom__" && (
              <div className="field">
                <label htmlFor="custom-model">Custom Model Name</label>
                <input
                  id="custom-model"
                  value={customModel}
                  onChange={(e) => setCustomModel(e.target.value)}
                  placeholder="e.g. llama3:70b"
                  required
                />
              </div>
            )}

            <button
              type="submit"
              className={`btn${isActive ? " running" : ""}`}
              disabled={isActive || !prompt.trim() || (selectedModel === "__custom__" && !customModel.trim())}
            >
              {isActive ? "Pipeline running..." : "Run SDLC Pipeline"}
            </button>
          </form>

          {submitError && (
            <div className="field-error">{submitError}</div>
          )}
        </section>

        {/* ── Right: live status ── */}
        <section className="panel">
          <p className="panel-title">Live Status</p>

          {!run ? (
            <p className="idle-hint">No active run. Start a pipeline to watch progress here.</p>
          ) : (
            <>
              <div className="kpi-row">
                <div className="kpi">
                  <div className="kpi-label">Status</div>
                  <div className={`kpi-value ${statusClass}`}>
                    {run.status.toUpperCase()}
                  </div>
                </div>
                <div className="kpi">
                  <div className="kpi-label">Progress</div>
                  <div className="kpi-value">{run.progress_percent}%</div>
                </div>
                <div className="kpi">
                  <div className="kpi-label">Active Agent</div>
                  <div className="kpi-value" style={{ fontSize: 13 }}>
                    {run.current_step_label || "—"}
                  </div>
                </div>
              </div>

              <div className="progress-wrap">
                <div
                  className="progress-bar"
                  style={{ width: `${run.progress_percent}%` }}
                />
              </div>

              <ol className="timeline">
                {run.steps.map((step) => (
                  <li key={step.key} className={`step-row ${step.status}`}>
                    <div className="step-indicator">
                      <span className="step-dot" />
                    </div>
                    <div className="step-body">
                      <div className="step-name">{step.label}</div>
                      <div className="step-status-label">{step.status}</div>
                    </div>
                  </li>
                ))}
              </ol>
            </>
          )}
        </section>

        {/* ── Bottom: outputs ── */}
        <section className="panel span-full">
          <p className="panel-title">Pipeline Outputs</p>

          <div className="tab-bar">
            {(
              [
                { key: "requirements", label: "Requirements" },
                { key: "code",         label: "Generated Code" },
                { key: "tests",        label: "Test Results" },
                { key: "review",       label: "Review Report" },
              ] as { key: ActiveTab; label: string }[]
            ).map(({ key, label }) => (
              <button
                key={key}
                className={`tab-btn${tab === key ? " active" : ""}`}
                onClick={() => setTab(key)}
              >
                {label}
              </button>
            ))}
          </div>

          {!run?.result ? (
            <p className="output-empty">
              Outputs appear here once the pipeline completes.
            </p>
          ) : (
            <div className="output-pane">
              {tab === "requirements" && (
                <pre>{prettyJson(run.result.requirements)}</pre>
              )}
              {tab === "code" && (
                <pre>{run.result.generated_code ?? "No code generated."}</pre>
              )}
              {tab === "tests" && (
                <pre>{prettyJson(run.result.test_results)}</pre>
              )}
              {tab === "review" && (
                <pre>{run.result.review_report ?? "No report generated."}</pre>
              )}
            </div>
          )}

          {(run?.errors?.length ?? 0) > 0 && (
            <div className="errors-panel">
              <p className="errors-panel-title">Errors & Warnings</p>
              <ul>
                {run!.errors.map((msg, i) => (
                  <li key={i}>{msg}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
