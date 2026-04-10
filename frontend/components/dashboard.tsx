"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { createRun, getRun, RunDetails, PipelineStep } from "@/lib/api";

type ActiveTab = "requirements" | "code" | "tests" | "review";

type ModelOption = {
  value: string;
  label: string;
  tier: "light" | "medium" | "heavy";
  note: string;
};

const MODEL_OPTIONS: ModelOption[] = [
  { value: "tinyllama",    label: "TinyLlama 1.1B",  tier: "light",  note: "~600 MB — fastest, great for dev" },
  { value: "qwen2:0.5b",  label: "Qwen2 0.5B",      tier: "light",  note: "~350 MB — ultra-light" },
  { value: "gemma:2b",    label: "Gemma 2B",         tier: "light",  note: "~1.4 GB — lightweight" },
  { value: "phi3:mini",   label: "Phi-3 Mini 3.8B",  tier: "light",  note: "~2.2 GB — smart & fast" },
  { value: "mistral:7b",  label: "Mistral 7B",       tier: "medium", note: "~4.1 GB — balanced" },
  { value: "qwen:7b",     label: "Qwen 7B",          tier: "medium", note: "~4.5 GB — balanced" },
  { value: "llama3:8b",   label: "Llama 3 8B",       tier: "heavy",  note: "~4.7 GB — recommended" },
  { value: "codellama:7b",label: "Code Llama 7B",    tier: "heavy",  note: "~3.8 GB — code-focused" },
  { value: "__custom__",  label: "Custom model...",  tier: "light",  note: "Enter a model name manually" },
];

const TIER_STYLES: Record<ModelOption["tier"], string> = {
  light:  "text-green-600 border-green-400 bg-green-50",
  medium: "text-amber-600 border-amber-400 bg-amber-50",
  heavy:  "text-stone-500 border-stone-300 bg-stone-50",
};

const STEP_STATUS_STYLES: Record<PipelineStep["status"], string> = {
  pending:   "bg-stone-200",
  running:   "bg-stone-800 shadow-[0_0_0_4px_rgba(17,17,17,0.1)]",
  completed: "bg-green-500",
  failed:    "bg-red-500",
};

const STEP_LABEL_STYLES: Record<PipelineStep["status"], string> = {
  pending:   "text-stone-400",
  running:   "text-stone-500",
  completed: "text-green-600",
  failed:    "text-red-600",
};

const RUN_STATUS_STYLES: Record<string, string> = {
  queued:    "text-stone-400",
  running:   "text-stone-700",
  completed: "text-green-600",
  failed:    "text-red-600",
};

const TABS: { key: ActiveTab; label: string }[] = [
  { key: "requirements", label: "Requirements" },
  { key: "code",         label: "Generated Code" },
  { key: "tests",        label: "Test Results" },
  { key: "review",       label: "Review Report" },
];

const DEFAULT_PROMPT =
  "Build a password reset module with secure token generation and expiry validation.";

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

/* ── Small reusable pieces ───────────────────────────────────────────────── */

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-bold tracking-[0.16em] uppercase text-stone-400 mb-5">
      {children}
    </p>
  );
}

function FieldLabel({ htmlFor, children }: { htmlFor?: string; children: React.ReactNode }) {
  return (
    <label
      htmlFor={htmlFor}
      className="block text-[11px] font-semibold tracking-[0.07em] uppercase text-stone-500 mb-2"
    >
      {children}
    </label>
  );
}

function inputClass() {
  return "w-full bg-stone-50 border border-stone-200 rounded-lg text-sm text-stone-900 placeholder:text-stone-400 px-3.5 py-2.5 focus:outline-none focus:border-stone-400 transition-colors font-sans appearance-none";
}

/* ── Main component ──────────────────────────────────────────────────────── */

export function Dashboard() {
  const [prompt, setPrompt]             = useState(DEFAULT_PROMPT);
  const [selectedModel, setSelectedModel] = useState("phi3:mini");
  const [customModel, setCustomModel]   = useState("");
  const [ollamaUrl, setOllamaUrl]       = useState("http://localhost:11434");
  const [runId, setRunId]               = useState<string | null>(null);
  const [run, setRun]                   = useState<RunDetails | null>(null);
  const [tab, setTab]                   = useState<ActiveTab>("requirements");
  const [submitError, setSubmitError]   = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const resolvedModel =
    selectedModel === "__custom__" ? customModel.trim() : selectedModel;

  const selectedOption = MODEL_OPTIONS.find((m) => m.value === selectedModel);

  /* polling */
  useEffect(() => {
    if (!runId) return;
    let active = true;
    const poll = async () => {
      try {
        const d = await getRun(runId);
        if (active) setRun(d);
      } catch (e) {
        if (active) setSubmitError((e as Error).message);
      }
    };
    poll();
    const id = setInterval(poll, 1500);
    return () => { active = false; clearInterval(id); };
  }, [runId]);

  const isActive = isSubmitting || run?.status === "queued" || run?.status === "running";

  const progressPct = run?.progress_percent ?? 0;

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSubmitError("");
    setIsSubmitting(true);
    setRun(null);
    try {
      const s = await createRun({ user_prompt: prompt.trim(), model_name: resolvedModel, ollama_base_url: ollamaUrl.trim() });
      setRunId(s.run_id);
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const canSubmit = !isActive && !!prompt.trim() && (selectedModel !== "__custom__" || !!customModel.trim());

  /* ── Render ── */
  return (
    <div className="max-w-screen-xl mx-auto px-6 pb-20">

      {/* Header */}
      <header className="py-14 border-b border-stone-100 mb-12">
        <p className="text-[10px] font-bold tracking-[0.2em] uppercase text-stone-400 mb-4">
          MAS SDLC — Multi-Agent Pipeline
        </p>
        <h1 className="text-6xl sm:text-7xl lg:text-8xl font-extrabold tracking-tighter leading-none text-stone-900">
          AI Software<br />Delivery Team
        </h1>
        <p className="mt-5 max-w-xl text-stone-500 text-[15px] leading-relaxed">
          Four specialised agents work sequentially to turn a plain-English
          feature request into requirements, working code, tests, and a review —
          all running locally via Ollama.
        </p>
      </header>

      {/* Workspace grid */}
      <div className="grid grid-cols-1 lg:grid-cols-[480px_1fr] gap-5">

        {/* ── Left: form ── */}
        <section className="bg-white border border-stone-200 rounded-xl p-7">
          <SectionLabel>New Pipeline Run</SectionLabel>

          <form onSubmit={onSubmit} className="space-y-5">
            {/* Prompt */}
            <div>
              <FieldLabel htmlFor="prompt">Feature Request</FieldLabel>
              <textarea
                id="prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                minLength={5}
                required
                rows={6}
                placeholder="Describe the feature you want to build..."
                className={inputClass() + " resize-y min-h-[120px]"}
              />
            </div>

            {/* Model + URL */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <FieldLabel htmlFor="model">Model</FieldLabel>
                <select
                  id="model"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className={inputClass()}
                >
                  {MODEL_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
                {selectedOption && selectedOption.value !== "__custom__" && (
                  <div className="flex items-center gap-2 mt-2">
                    <span className={`text-[9px] font-bold tracking-widest uppercase px-1.5 py-0.5 rounded border ${TIER_STYLES[selectedOption.tier]}`}>
                      {selectedOption.tier}
                    </span>
                    <span className="text-[11px] text-stone-400">{selectedOption.note}</span>
                  </div>
                )}
              </div>

              <div>
                <FieldLabel htmlFor="ollama-url">Ollama URL</FieldLabel>
                <input
                  id="ollama-url"
                  value={ollamaUrl}
                  onChange={(e) => setOllamaUrl(e.target.value)}
                  required
                  className={inputClass()}
                />
              </div>
            </div>

            {/* Custom model input */}
            {selectedModel === "__custom__" && (
              <div>
                <FieldLabel htmlFor="custom-model">Custom Model Name</FieldLabel>
                <input
                  id="custom-model"
                  value={customModel}
                  onChange={(e) => setCustomModel(e.target.value)}
                  placeholder="e.g. llama3:70b"
                  required
                  className={inputClass()}
                />
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={!canSubmit}
              className={`w-full py-3 rounded-lg text-sm font-bold tracking-wide transition-all ${
                isActive
                  ? "bg-stone-100 text-stone-400 border border-stone-200 cursor-not-allowed"
                  : "bg-stone-900 text-white hover:bg-stone-700 active:scale-[0.98] cursor-pointer"
              } disabled:opacity-40 disabled:cursor-not-allowed`}
            >
              {isActive ? "Pipeline running..." : "Run SDLC Pipeline"}
            </button>
          </form>

          {submitError && (
            <div className="mt-4 px-4 py-3 border border-red-200 bg-red-50 rounded-lg text-red-600 text-sm">
              {submitError}
            </div>
          )}
        </section>

        {/* ── Right: live status ── */}
        <section className="bg-white border border-stone-200 rounded-xl p-7">
          <SectionLabel>Live Status</SectionLabel>

          {!run ? (
            <p className="text-stone-400 text-sm">No active run. Start a pipeline to watch progress here.</p>
          ) : (
            <>
              {/* KPI row */}
              <div className="grid grid-cols-3 gap-3 mb-6">
                {[
                  { label: "Status",       value: run.status.toUpperCase(), extra: RUN_STATUS_STYLES[run.status] },
                  { label: "Progress",     value: `${progressPct}%`,         extra: "text-stone-900" },
                  { label: "Active Agent", value: run.current_step_label ?? "—", extra: "text-stone-900 text-xs" },
                ].map(({ label, value, extra }) => (
                  <div key={label} className="border border-stone-100 rounded-lg p-4">
                    <p className="text-[10px] font-bold tracking-[0.14em] uppercase text-stone-400 mb-1.5">{label}</p>
                    <p className={`font-bold text-sm leading-tight ${extra}`}>{value}</p>
                  </div>
                ))}
              </div>

              {/* Progress bar */}
              <div className="h-[2px] bg-stone-100 rounded-full overflow-hidden mb-7">
                <div
                  className="h-full bg-stone-900 rounded-full transition-all duration-500"
                  style={{ width: `${progressPct}%` }}
                />
              </div>

              {/* Timeline */}
              <ol className="space-y-0">
                {run.steps.map((step, i) => (
                  <li
                    key={step.key}
                    className={`flex items-start gap-4 py-3.5 ${i < run.steps.length - 1 ? "border-b border-stone-50" : ""}`}
                  >
                    <span className={`mt-1 w-2.5 h-2.5 rounded-full shrink-0 ${STEP_STATUS_STYLES[step.status]}`} />
                    <div>
                      <p className={`text-sm font-semibold ${step.status === "pending" ? "text-stone-400" : "text-stone-900"}`}>
                        {step.label}
                      </p>
                      <p className={`text-[11px] font-semibold tracking-widest uppercase mt-0.5 ${STEP_LABEL_STYLES[step.status]}`}>
                        {step.status}
                      </p>
                    </div>
                  </li>
                ))}
              </ol>
            </>
          )}
        </section>

        {/* ── Full-width: outputs ── */}
        <section className="lg:col-span-2 bg-white border border-stone-200 rounded-xl p-7">
          <SectionLabel>Pipeline Outputs</SectionLabel>

          {/* Tab bar */}
          <div className="flex border-b border-stone-100 mb-6 -mx-1">
            {TABS.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={`px-4 py-2.5 text-sm font-semibold tracking-wide border-b-2 transition-colors ${
                  tab === key
                    ? "border-stone-900 text-stone-900"
                    : "border-transparent text-stone-400 hover:text-stone-600"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {!run?.result ? (
            <p className="text-stone-400 text-sm py-4">Outputs appear here once the pipeline completes.</p>
          ) : (
            <pre className="bg-stone-50 border border-stone-100 rounded-lg p-5 max-h-[500px] overflow-auto text-[12.5px] leading-7 text-stone-700 font-mono whitespace-pre-wrap break-words">
              {tab === "requirements" && prettyJson(run.result.requirements)}
              {tab === "code"         && (run.result.generated_code      ?? "No code generated.")}
              {tab === "tests"        && prettyJson(run.result.test_results)}
              {tab === "review"       && (run.result.review_report        ?? "No report generated.")}
            </pre>
          )}

          {(run?.errors?.length ?? 0) > 0 && (
            <div className="mt-5 border border-red-200 rounded-lg p-5">
              <p className="text-[10px] font-bold tracking-[0.16em] uppercase text-red-500 mb-3">
                Errors & Warnings
              </p>
              <ul className="list-disc pl-4 space-y-1.5">
                {run!.errors.map((msg, i) => (
                  <li key={i} className="text-red-700 text-sm">{msg}</li>
                ))}
              </ul>
            </div>
          )}
        </section>

      </div>
    </div>
  );
}
